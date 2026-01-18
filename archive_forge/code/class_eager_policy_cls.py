import functools
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple, Union
import tree  # pip install dm_tree
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import add_mixins, force_list
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
class eager_policy_cls(base):

    def __init__(self, observation_space, action_space, config):
        if not tf1.executing_eagerly():
            tf1.enable_eager_execution()
        self.framework = config.get('framework', 'tf2')
        EagerTFPolicy.__init__(self, observation_space, action_space, config)
        self.global_timestep = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.explore = tf.Variable(self.config['explore'], trainable=False, dtype=tf.bool)
        num_gpus = self._get_num_gpus_for_policy()
        if num_gpus > 0:
            gpu_ids = get_gpu_devices()
            logger.info(f'Found {len(gpu_ids)} visible cuda devices.')
        self._is_training = False
        self._re_trace_counter = 0
        self._loss_initialized = False
        if loss_fn is not None:
            self._loss = loss_fn
        elif self.loss.__func__.__qualname__ != 'Policy.loss':
            self._loss = self.loss.__func__
        else:
            self._loss = None
        self.batch_divisibility_req = get_batch_divisibility_req(self) if callable(get_batch_divisibility_req) else get_batch_divisibility_req or 1
        self._max_seq_len = config['model']['max_seq_len']
        if validate_spaces:
            validate_spaces(self, observation_space, action_space, config)
        if before_init:
            before_init(self, observation_space, action_space, config)
        self.config = config
        self.dist_class = None
        if action_sampler_fn or action_distribution_fn:
            if not make_model:
                raise ValueError('`make_model` is required if `action_sampler_fn` OR `action_distribution_fn` is given')
        else:
            self.dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, self.config['model'])
        if make_model:
            self.model = make_model(self, observation_space, action_space, config)
        else:
            self.model = ModelCatalog.get_model_v2(observation_space, action_space, logit_dim, config['model'], framework=self.framework)
        self._lock = threading.RLock()
        if self.config.get('_enable_new_api_stack', False):
            self.view_requirements = self.model.update_default_view_requirements(self.view_requirements)
        else:
            self._update_model_view_requirements_from_init_state()
            self.view_requirements.update(self.model.view_requirements)
        self.exploration = self._create_exploration()
        self._state_inputs = self.model.get_initial_state()
        self._is_recurrent = len(self._state_inputs) > 0
        if before_loss_init:
            before_loss_init(self, observation_space, action_space, config)
        if optimizer_fn:
            optimizers = optimizer_fn(self, config)
        else:
            optimizers = tf.keras.optimizers.Adam(config['lr'])
        optimizers = force_list(optimizers)
        if self.exploration:
            optimizers = self.exploration.get_exploration_optimizer(optimizers)
        self._optimizers: List[LocalOptimizer] = optimizers
        self._optimizer: LocalOptimizer = optimizers[0] if optimizers else None
        self._initialize_loss_from_dummy_batch(auto_remove_unneeded_view_reqs=True, stats_fn=stats_fn)
        self._loss_initialized = True
        if after_init:
            after_init(self, observation_space, action_space, config)
        self.global_timestep.assign(0)

    @override(Policy)
    def compute_actions_from_input_dict(self, input_dict: Dict[str, TensorType], explore: bool=None, timestep: Optional[int]=None, episodes: Optional[List[Episode]]=None, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        if not self.config.get('eager_tracing') and (not tf1.executing_eagerly()):
            tf1.enable_eager_execution()
        self._is_training = False
        explore = explore if explore is not None else self.explore
        timestep = timestep if timestep is not None else self.global_timestep
        if isinstance(timestep, tf.Tensor):
            timestep = int(timestep.numpy())
        input_dict = self._lazy_tensor_dict(input_dict)
        input_dict.set_training(False)
        state_batches = [input_dict[k] for k in input_dict.keys() if 'state_in' in k[:8]]
        self._state_in = state_batches
        self._is_recurrent = state_batches != []
        self.exploration.before_compute_actions(timestep=timestep, explore=explore, tf_sess=self.get_session())
        ret = self._compute_actions_helper(input_dict, state_batches, None if self.config['eager_tracing'] else episodes, explore, timestep)
        self.global_timestep.assign_add(tree.flatten(ret[0])[0].shape.as_list()[0])
        return convert_to_numpy(ret)

    @override(Policy)
    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType], state_batches: Optional[List[TensorType]]=None, prev_action_batch: Union[List[TensorStructType], TensorStructType]=None, prev_reward_batch: Union[List[TensorStructType], TensorStructType]=None, info_batch: Optional[Dict[str, list]]=None, episodes: Optional[List['Episode']]=None, explore: Optional[bool]=None, timestep: Optional[int]=None, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        input_dict = SampleBatch({SampleBatch.CUR_OBS: obs_batch}, _is_training=tf.constant(False))
        if state_batches is not None:
            for i, s in enumerate(state_batches):
                input_dict[f'state_in_{i}'] = s
        if prev_action_batch is not None:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
        if info_batch is not None:
            input_dict[SampleBatch.INFOS] = info_batch
        return self.compute_actions_from_input_dict(input_dict=input_dict, explore=explore, timestep=timestep, episodes=episodes, **kwargs)

    @with_lock
    @override(Policy)
    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, actions_normalized=True, **kwargs):
        if action_sampler_fn and action_distribution_fn is None:
            raise ValueError('Cannot compute log-prob/likelihood w/o an `action_distribution_fn` and a provided `action_sampler_fn`!')
        seq_lens = tf.ones(len(obs_batch), dtype=tf.int32)
        input_batch = SampleBatch({SampleBatch.CUR_OBS: tf.convert_to_tensor(obs_batch)}, _is_training=False)
        if prev_action_batch is not None:
            input_batch[SampleBatch.PREV_ACTIONS] = tf.convert_to_tensor(prev_action_batch)
        if prev_reward_batch is not None:
            input_batch[SampleBatch.PREV_REWARDS] = tf.convert_to_tensor(prev_reward_batch)
        if self.exploration:
            self.exploration.before_compute_actions(explore=False)
        if action_distribution_fn:
            dist_inputs, dist_class, _ = action_distribution_fn(self, self.model, input_batch, explore=False, is_training=False)
        else:
            dist_inputs, _ = self.model(input_batch, state_batches, seq_lens)
            dist_class = self.dist_class
        action_dist = dist_class(dist_inputs, self.model)
        if not actions_normalized and self.config['normalize_actions']:
            actions = normalize_action(actions, self.action_space_struct)
        log_likelihoods = action_dist.logp(actions)
        return log_likelihoods

    @override(Policy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        assert tf.executing_eagerly()
        sample_batch = EagerTFPolicy.postprocess_trajectory(self, sample_batch)
        if postprocess_fn:
            return postprocess_fn(self, sample_batch, other_agent_batches, episode)
        return sample_batch

    @with_lock
    @override(Policy)
    def learn_on_batch(self, postprocessed_batch):
        learn_stats = {}
        self.callbacks.on_learn_on_batch(policy=self, train_batch=postprocessed_batch, result=learn_stats)
        pad_batch_to_sequences_of_same_size(postprocessed_batch, max_seq_len=self._max_seq_len, shuffle=False, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements)
        self._is_training = True
        postprocessed_batch = self._lazy_tensor_dict(postprocessed_batch)
        postprocessed_batch.set_training(True)
        stats = self._learn_on_batch_helper(postprocessed_batch)
        self.num_grad_updates += 1
        stats.update({'custom_metrics': learn_stats, NUM_AGENT_STEPS_TRAINED: postprocessed_batch.count, NUM_GRAD_UPDATES_LIFETIME: self.num_grad_updates, DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY: self.num_grad_updates - 1 - (postprocessed_batch.num_grad_updates or 0)})
        return convert_to_numpy(stats)

    @override(Policy)
    def compute_gradients(self, postprocessed_batch: SampleBatch) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        pad_batch_to_sequences_of_same_size(postprocessed_batch, shuffle=False, max_seq_len=self._max_seq_len, batch_divisibility_req=self.batch_divisibility_req, view_requirements=self.view_requirements)
        self._is_training = True
        self._lazy_tensor_dict(postprocessed_batch)
        postprocessed_batch.set_training(True)
        grads_and_vars, grads, stats = self._compute_gradients_helper(postprocessed_batch)
        return convert_to_numpy((grads, stats))

    @override(Policy)
    def apply_gradients(self, gradients: ModelGradients) -> None:
        self._apply_gradients_helper(list(zip([tf.convert_to_tensor(g) if g is not None else None for g in gradients], self.model.trainable_variables())))

    @override(Policy)
    def get_weights(self, as_dict=False):
        variables = self.variables()
        if as_dict:
            return {v.name: v.numpy() for v in variables}
        return [v.numpy() for v in variables]

    @override(Policy)
    def set_weights(self, weights):
        variables = self.variables()
        assert len(weights) == len(variables), (len(weights), len(variables))
        for v, w in zip(variables, weights):
            v.assign(w)

    @override(Policy)
    def get_exploration_state(self):
        return convert_to_numpy(self.exploration.get_state())

    @override(Policy)
    def is_recurrent(self):
        return self._is_recurrent

    @override(Policy)
    def num_state_tensors(self):
        return len(self._state_inputs)

    @override(Policy)
    def get_initial_state(self):
        if hasattr(self, 'model'):
            return self.model.get_initial_state()
        return []

    @override(Policy)
    def get_state(self) -> PolicyState:
        state = super().get_state()
        state['global_timestep'] = state['global_timestep'].numpy()
        if self._optimizer and len(self._optimizer.variables()) > 0:
            state['_optimizer_variables'] = self._optimizer.variables()
        if not self.config.get('_enable_new_api_stack', False) and self.exploration:
            state['_exploration_state'] = self.exploration.get_state()
        return state

    @override(Policy)
    def set_state(self, state: PolicyState) -> None:
        optimizer_vars = state.get('_optimizer_variables', None)
        if optimizer_vars and self._optimizer.variables():
            if not type(self).__name__.endswith('_traced') and log_once('set_state_optimizer_vars_tf_eager_policy_v2'):
                logger.warning("Cannot restore an optimizer's state for tf eager! Keras is not able to save the v1.x optimizers (from tf.compat.v1.train) since they aren't compatible with checkpoints.")
            for opt_var, value in zip(self._optimizer.variables(), optimizer_vars):
                opt_var.assign(value)
        if hasattr(self, 'exploration') and '_exploration_state' in state:
            self.exploration.set_state(state=state['_exploration_state'])
        self.global_timestep.assign(state['global_timestep'])
        super().set_state(state)

    @override(Policy)
    def export_model(self, export_dir, onnx: Optional[int]=None) -> None:
        """Exports the Policy's Model to local directory for serving.

            Note: Since the TfModelV2 class that EagerTfPolicy uses is-NOT-a
            tf.keras.Model, we need to assume that there is a `base_model` property
            within this TfModelV2 class that is-a tf.keras.Model. This base model
            will be used here for the export.
            TODO (kourosh): This restriction will be resolved once we move Policy and
            ModelV2 to the new Learner/RLModule APIs.

            Args:
                export_dir: Local writable directory.
                onnx: If given, will export model in ONNX format. The
                    value of this parameter set the ONNX OpSet version to use.
            """
        if hasattr(self, 'model') and hasattr(self.model, 'base_model') and isinstance(self.model.base_model, tf.keras.Model):
            if onnx:
                try:
                    import tf2onnx
                except ImportError as e:
                    raise RuntimeError('Converting a TensorFlow model to ONNX requires `tf2onnx` to be installed. Install with `pip install tf2onnx`.') from e
                model_proto, external_tensor_storage = tf2onnx.convert.from_keras(self.model.base_model, output_path=os.path.join(export_dir, 'model.onnx'))
            else:
                try:
                    self.model.base_model.save(export_dir, save_format='tf')
                except Exception:
                    logger.warning(ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL)
        else:
            logger.warning(ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL)

    def variables(self):
        """Return the list of all savable variables for this policy."""
        if isinstance(self.model, tf.keras.Model):
            return self.model.variables
        else:
            return self.model.variables()

    def loss_initialized(self):
        return self._loss_initialized

    @with_lock
    def _compute_actions_helper(self, input_dict, state_batches, episodes, explore, timestep):
        self._re_trace_counter += 1
        batch_size = tree.flatten(input_dict[SampleBatch.OBS])[0].shape[0]
        seq_lens = tf.ones(batch_size, dtype=tf.int32) if state_batches else None
        extra_fetches = {}
        with tf.variable_creator_scope(_disallow_var_creation):
            if action_sampler_fn:
                action_sampler_outputs = action_sampler_fn(self, self.model, input_dict[SampleBatch.CUR_OBS], explore=explore, timestep=timestep, episodes=episodes)
                if len(action_sampler_outputs) == 4:
                    actions, logp, dist_inputs, state_out = action_sampler_outputs
                else:
                    dist_inputs = None
                    state_out = []
                    actions, logp = action_sampler_outputs
            else:
                if action_distribution_fn:
                    try:
                        dist_inputs, self.dist_class, state_out = action_distribution_fn(self, self.model, input_dict=input_dict, state_batches=state_batches, seq_lens=seq_lens, explore=explore, timestep=timestep, is_training=False)
                    except TypeError as e:
                        if 'positional argument' in e.args[0] or 'unexpected keyword argument' in e.args[0]:
                            dist_inputs, self.dist_class, state_out = action_distribution_fn(self, self.model, input_dict[SampleBatch.OBS], explore=explore, timestep=timestep, is_training=False)
                        else:
                            raise e
                elif isinstance(self.model, tf.keras.Model):
                    input_dict = SampleBatch(input_dict, seq_lens=seq_lens)
                    if state_batches and 'state_in_0' not in input_dict:
                        for i, s in enumerate(state_batches):
                            input_dict[f'state_in_{i}'] = s
                    self._lazy_tensor_dict(input_dict)
                    dist_inputs, state_out, extra_fetches = self.model(input_dict)
                else:
                    dist_inputs, state_out = self.model(input_dict, state_batches, seq_lens)
                action_dist = self.dist_class(dist_inputs, self.model)
                actions, logp = self.exploration.get_exploration_action(action_distribution=action_dist, timestep=timestep, explore=explore)
        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = tf.exp(logp)
            extra_fetches[SampleBatch.ACTION_LOGP] = logp
        if dist_inputs is not None:
            extra_fetches[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs
        if extra_action_out_fn:
            extra_fetches.update(extra_action_out_fn(self))
        return (actions, state_out, extra_fetches)

    def _learn_on_batch_helper(self, samples, _ray_trace_ctx=None):
        self._re_trace_counter += 1
        with tf.variable_creator_scope(_disallow_var_creation):
            grads_and_vars, _, stats = self._compute_gradients_helper(samples)
        self._apply_gradients_helper(grads_and_vars)
        return stats

    def _get_is_training_placeholder(self):
        return tf.convert_to_tensor(self._is_training)

    @with_lock
    def _compute_gradients_helper(self, samples):
        """Computes and returns grads as eager tensors."""
        self._re_trace_counter += 1
        if isinstance(self.model, tf.keras.Model):
            variables = self.model.trainable_variables
        else:
            variables = self.model.trainable_variables()
        with tf.GradientTape(persistent=compute_gradients_fn is not None) as tape:
            losses = self._loss(self, self.model, self.dist_class, samples)
        losses = force_list(losses)
        if compute_gradients_fn:
            optimizer = _OptimizerWrapper(tape)
            if self.config['_tf_policy_handles_more_than_one_loss']:
                grads_and_vars = compute_gradients_fn(self, [optimizer] * len(losses), losses)
            else:
                grads_and_vars = [compute_gradients_fn(self, optimizer, losses[0])]
        else:
            grads_and_vars = [list(zip(tape.gradient(loss, variables), variables)) for loss in losses]
        if log_once('grad_vars'):
            for g_and_v in grads_and_vars:
                for g, v in g_and_v:
                    if g is not None:
                        logger.info(f'Optimizing variable {v.name}')
        if self.config['_tf_policy_handles_more_than_one_loss']:
            grads = [[g for g, _ in g_and_v] for g_and_v in grads_and_vars]
        else:
            grads_and_vars = grads_and_vars[0]
            grads = [g for g, _ in grads_and_vars]
        stats = self._stats(self, samples, grads)
        return (grads_and_vars, grads, stats)

    def _apply_gradients_helper(self, grads_and_vars):
        self._re_trace_counter += 1
        if apply_gradients_fn:
            if self.config['_tf_policy_handles_more_than_one_loss']:
                apply_gradients_fn(self, self._optimizers, grads_and_vars)
            else:
                apply_gradients_fn(self, self._optimizer, grads_and_vars)
        elif self.config['_tf_policy_handles_more_than_one_loss']:
            for i, o in enumerate(self._optimizers):
                o.apply_gradients([(g, v) for g, v in grads_and_vars[i] if g is not None])
        else:
            self._optimizer.apply_gradients([(g, v) for g, v in grads_and_vars if g is not None])

    def _stats(self, outputs, samples, grads):
        fetches = {}
        if stats_fn:
            fetches[LEARNER_STATS_KEY] = {k: v for k, v in stats_fn(outputs, samples).items()}
        else:
            fetches[LEARNER_STATS_KEY] = {}
        if extra_learn_fetches_fn:
            fetches.update({k: v for k, v in extra_learn_fetches_fn(self).items()})
        if grad_stats_fn:
            fetches.update({k: v for k, v in grad_stats_fn(self, samples, grads).items()})
        return fetches

    def _lazy_tensor_dict(self, postprocessed_batch: SampleBatch):
        if not isinstance(postprocessed_batch, SampleBatch):
            postprocessed_batch = SampleBatch(postprocessed_batch)
        postprocessed_batch.set_get_interceptor(_convert_to_tf)
        return postprocessed_batch

    @classmethod
    def with_tracing(cls):
        return _traced_eager_policy(cls)