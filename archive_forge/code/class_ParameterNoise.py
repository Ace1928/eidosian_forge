from gymnasium.spaces import Box, Discrete
import numpy as np
from typing import Optional, TYPE_CHECKING, Union
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, Deterministic
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import softmax, SMALL_NUMBER
from ray.rllib.utils.typing import TensorType
@PublicAPI
class ParameterNoise(Exploration):
    """An exploration that changes a Model's parameters.

    Implemented based on:
    [1] https://openai.com/research/better-exploration-with-parameter-noise
    [2] https://arxiv.org/pdf/1706.01905.pdf

    At the beginning of an episode, Gaussian noise is added to all weights
    of the model. At the end of the episode, the noise is undone and an action
    diff (pi-delta) is calculated, from which we determine the changes in the
    noise's stddev for the next episode.
    """

    def __init__(self, action_space, *, framework: str, policy_config: dict, model: ModelV2, initial_stddev: float=1.0, random_timesteps: int=10000, sub_exploration: Optional[dict]=None, **kwargs):
        """Initializes a ParameterNoise Exploration object.

        Args:
            initial_stddev: The initial stddev to use for the noise.
            random_timesteps: The number of timesteps to act completely
                randomly (see [1]).
            sub_exploration: Optional sub-exploration config.
                None for auto-detection/setup.
        """
        assert framework is not None
        super().__init__(action_space, policy_config=policy_config, model=model, framework=framework, **kwargs)
        self.stddev = get_variable(initial_stddev, framework=self.framework, tf_name='stddev')
        self.stddev_val = initial_stddev
        self.model_variables = [v for k, v in self.model.trainable_variables(as_dict=True).items() if 'LayerNorm' not in k]
        self.noise = []
        for var in self.model_variables:
            name_ = var.name.split(':')[0] + '_noisy' if var.name else ''
            self.noise.append(get_variable(np.zeros(var.shape, dtype=np.float32), framework=self.framework, tf_name=name_, torch_tensor=True, device=self.device))
        if self.framework == 'tf' and (not tf.executing_eagerly()):
            self.tf_sample_new_noise_op = self._tf_sample_new_noise_op()
            self.tf_add_stored_noise_op = self._tf_add_stored_noise_op()
            self.tf_remove_noise_op = self._tf_remove_noise_op()
            with tf1.control_dependencies([self.tf_sample_new_noise_op]):
                add_op = self._tf_add_stored_noise_op()
            with tf1.control_dependencies([add_op]):
                self.tf_sample_new_noise_and_add_op = tf.no_op()
        self.weights_are_currently_noisy = False
        if sub_exploration is None:
            if isinstance(self.action_space, Discrete):
                sub_exploration = {'type': 'EpsilonGreedy', 'epsilon_schedule': {'type': 'PiecewiseSchedule', 'endpoints': [(0, 1.0), (random_timesteps + 1, 1.0), (random_timesteps + 2, 0.01)], 'outside_value': 0.01}}
            elif isinstance(self.action_space, Box):
                sub_exploration = {'type': 'OrnsteinUhlenbeckNoise', 'random_timesteps': random_timesteps}
            else:
                raise NotImplementedError
        self.sub_exploration = from_config(Exploration, sub_exploration, framework=self.framework, action_space=self.action_space, policy_config=self.policy_config, model=self.model, **kwargs)
        self.episode_started = False

    @override(Exploration)
    def before_compute_actions(self, *, timestep: Optional[int]=None, explore: Optional[bool]=None, tf_sess: Optional['tf.Session']=None):
        explore = explore if explore is not None else self.policy_config['explore']
        if self.episode_started:
            self._delayed_on_episode_start(explore, tf_sess)
        if explore and (not self.weights_are_currently_noisy):
            self._add_stored_noise(tf_sess=tf_sess)
        elif not explore and self.weights_are_currently_noisy:
            self._remove_noise(tf_sess=tf_sess)

    @override(Exploration)
    def get_exploration_action(self, *, action_distribution: ActionDistribution, timestep: Union[TensorType, int], explore: Union[TensorType, bool]):
        return self.sub_exploration.get_exploration_action(action_distribution=action_distribution, timestep=timestep, explore=explore)

    @override(Exploration)
    def on_episode_start(self, policy: 'Policy', *, environment: BaseEnv=None, episode: int=None, tf_sess: Optional['tf.Session']=None):
        self.episode_started = True

    def _delayed_on_episode_start(self, explore, tf_sess):
        if explore:
            self._sample_new_noise_and_add(tf_sess=tf_sess, override=True)
        else:
            self._sample_new_noise(tf_sess=tf_sess)
        self.episode_started = False

    @override(Exploration)
    def on_episode_end(self, policy, *, environment=None, episode=None, tf_sess=None):
        if self.weights_are_currently_noisy:
            self._remove_noise(tf_sess=tf_sess)

    @override(Exploration)
    def postprocess_trajectory(self, policy: 'Policy', sample_batch: SampleBatch, tf_sess: Optional['tf.Session']=None):
        noisy_action_dist = noise_free_action_dist = None
        _, _, fetches = policy.compute_actions_from_input_dict(input_dict=sample_batch, explore=self.weights_are_currently_noisy)
        if issubclass(policy.dist_class, (Categorical, TorchCategorical)):
            action_dist = softmax(fetches[SampleBatch.ACTION_DIST_INPUTS])
        elif issubclass(policy.dist_class, (Deterministic, TorchDeterministic)):
            action_dist = fetches[SampleBatch.ACTION_DIST_INPUTS]
        else:
            raise NotImplementedError
        if self.weights_are_currently_noisy:
            noisy_action_dist = action_dist
        else:
            noise_free_action_dist = action_dist
        _, _, fetches = policy.compute_actions_from_input_dict(input_dict=sample_batch, explore=not self.weights_are_currently_noisy)
        if issubclass(policy.dist_class, (Categorical, TorchCategorical)):
            action_dist = softmax(fetches[SampleBatch.ACTION_DIST_INPUTS])
        elif issubclass(policy.dist_class, (Deterministic, TorchDeterministic)):
            action_dist = fetches[SampleBatch.ACTION_DIST_INPUTS]
        if noisy_action_dist is None:
            noisy_action_dist = action_dist
        else:
            noise_free_action_dist = action_dist
        delta = distance = None
        if issubclass(policy.dist_class, (Categorical, TorchCategorical)):
            distance = np.nanmean(np.sum(noise_free_action_dist * np.log(noise_free_action_dist / (noisy_action_dist + SMALL_NUMBER)), 1))
            current_epsilon = self.sub_exploration.get_state(sess=tf_sess)['cur_epsilon']
            delta = -np.log(1 - current_epsilon + current_epsilon / self.action_space.n)
        elif issubclass(policy.dist_class, (Deterministic, TorchDeterministic)):
            distance = np.sqrt(np.mean(np.square(noise_free_action_dist - noisy_action_dist)))
            current_scale = self.sub_exploration.get_state(sess=tf_sess)['cur_scale']
            delta = getattr(self.sub_exploration, 'ou_sigma', 0.2) * current_scale
        if distance <= delta:
            self.stddev_val *= 1.01
        else:
            self.stddev_val /= 1.01
        self.set_state(self.get_state(), sess=tf_sess)
        return sample_batch

    def _sample_new_noise(self, *, tf_sess=None):
        """Samples new noise and stores it in `self.noise`."""
        if self.framework == 'tf':
            tf_sess.run(self.tf_sample_new_noise_op)
        elif self.framework == 'tf2':
            self._tf_sample_new_noise_op()
        else:
            for i in range(len(self.noise)):
                self.noise[i] = torch.normal(mean=torch.zeros(self.noise[i].size()), std=self.stddev).to(self.device)

    def _tf_sample_new_noise_op(self):
        added_noises = []
        for noise in self.noise:
            added_noises.append(tf1.assign(noise, tf.random.normal(shape=noise.shape, stddev=self.stddev, dtype=tf.float32)))
        return tf.group(*added_noises)

    def _sample_new_noise_and_add(self, *, tf_sess=None, override=False):
        if self.framework == 'tf':
            if override and self.weights_are_currently_noisy:
                tf_sess.run(self.tf_remove_noise_op)
            tf_sess.run(self.tf_sample_new_noise_and_add_op)
        else:
            if override and self.weights_are_currently_noisy:
                self._remove_noise()
            self._sample_new_noise()
            self._add_stored_noise()
        self.weights_are_currently_noisy = True

    def _add_stored_noise(self, *, tf_sess=None):
        """Adds the stored `self.noise` to the model's parameters.

        Note: No new sampling of noise here.

        Args:
            tf_sess (Optional[tf.Session]): The tf-session to use to add the
                stored noise to the (currently noise-free) weights.
            override: If True, undo any currently applied noise first,
                then add the currently stored noise.
        """
        assert self.weights_are_currently_noisy is False
        if self.framework == 'tf':
            tf_sess.run(self.tf_add_stored_noise_op)
        elif self.framework == 'tf2':
            self._tf_add_stored_noise_op()
        else:
            for var, noise in zip(self.model_variables, self.noise):
                var.requires_grad = False
                var.add_(noise)
                var.requires_grad = True
        self.weights_are_currently_noisy = True

    def _tf_add_stored_noise_op(self):
        """Generates tf-op that assigns the stored noise to weights.

        Also used by tf-eager.

        Returns:
            tf.op: The tf op to apply the already stored noise to the NN.
        """
        add_noise_ops = list()
        for var, noise in zip(self.model_variables, self.noise):
            add_noise_ops.append(tf1.assign_add(var, noise))
        ret = tf.group(*tuple(add_noise_ops))
        with tf1.control_dependencies([ret]):
            return tf.no_op()

    def _remove_noise(self, *, tf_sess=None):
        """
        Removes the current action noise from the model parameters.

        Args:
            tf_sess (Optional[tf.Session]): The tf-session to use to remove
                the noise from the (currently noisy) weights.
        """
        assert self.weights_are_currently_noisy is True
        if self.framework == 'tf':
            tf_sess.run(self.tf_remove_noise_op)
        elif self.framework == 'tf2':
            self._tf_remove_noise_op()
        else:
            for var, noise in zip(self.model_variables, self.noise):
                var.requires_grad = False
                var.add_(-noise)
                var.requires_grad = True
        self.weights_are_currently_noisy = False

    def _tf_remove_noise_op(self):
        """Generates a tf-op for removing noise from the model's weights.

        Also used by tf-eager.

        Returns:
            tf.op: The tf op to remve the currently stored noise from the NN.
        """
        remove_noise_ops = list()
        for var, noise in zip(self.model_variables, self.noise):
            remove_noise_ops.append(tf1.assign_add(var, -noise))
        ret = tf.group(*tuple(remove_noise_ops))
        with tf1.control_dependencies([ret]):
            return tf.no_op()

    @override(Exploration)
    def get_state(self, sess=None):
        return {'cur_stddev': self.stddev_val}

    @override(Exploration)
    def set_state(self, state: dict, sess: Optional['tf.Session']=None) -> None:
        self.stddev_val = state['cur_stddev']
        if self.framework == 'tf':
            self.stddev.load(self.stddev_val, session=sess)
        elif isinstance(self.stddev, float):
            self.stddev = self.stddev_val
        else:
            self.stddev.assign(self.stddev_val)