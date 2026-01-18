import dataclasses
from typing import Optional, Type
import logging
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.appo.appo_learner import (
from ray.rllib.algorithms.impala.impala import Impala, ImpalaConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics import ALL_MODULES, LEARNER_STATS_KEY
from ray.rllib.utils.typing import (
class APPOConfig(ImpalaConfig):
    """Defines a configuration class from which an APPO Algorithm can be built.

    .. testcode::

        from ray.rllib.algorithms.appo import APPOConfig
        config = APPOConfig().training(lr=0.01, grad_clip=30.0, train_batch_size=50)
        config = config.resources(num_gpus=0)
        config = config.rollouts(num_rollout_workers=1)
        config = config.environment("CartPole-v1")

        # Build an Algorithm object from the config and run 1 training iteration.
        algo = config.build()
        algo.train()
        del algo

    .. testcode::

        from ray.rllib.algorithms.appo import APPOConfig
        from ray import air
        from ray import tune

        config = APPOConfig()
        # Update the config object.
        config = config.training(lr=tune.grid_search([0.001,]))
        # Set the config object's env.
        config = config.environment(env="CartPole-v1")
        # Use to_dict() to get the old-style python config dict
        # when running with tune.
        tune.Tuner(
            "APPO",
            run_config=air.RunConfig(stop={"training_iteration": 1},
                                     verbose=0),
            param_space=config.to_dict(),

        ).fit()

    .. testoutput::
        :hide:

        ...
    """

    def __init__(self, algo_class=None):
        """Initializes a APPOConfig instance."""
        super().__init__(algo_class=algo_class or APPO)
        self.vtrace = True
        self.use_critic = True
        self.use_gae = True
        self.lambda_ = 1.0
        self.clip_param = 0.4
        self.use_kl_loss = False
        self.kl_coeff = 1.0
        self.kl_target = 0.01
        self.num_rollout_workers = 2
        self.rollout_fragment_length = 50
        self.train_batch_size = 500
        self.min_time_s_per_iteration = 10
        self.num_gpus = 0
        self.num_multi_gpu_tower_stacks = 1
        self.minibatch_buffer_size = 1
        self.num_sgd_iter = 1
        self.target_update_frequency = 1
        self.replay_proportion = 0.0
        self.replay_buffer_num_slots = 100
        self.learner_queue_size = 16
        self.learner_queue_timeout = 300
        self.max_sample_requests_in_flight_per_worker = 2
        self.broadcast_interval = 1
        self.grad_clip = 40.0
        self.grad_clip_by = 'global_norm'
        self.opt_type = 'adam'
        self.lr = 0.0005
        self.lr_schedule = None
        self.decay = 0.99
        self.momentum = 0.0
        self.epsilon = 0.1
        self.vf_loss_coeff = 0.5
        self.entropy_coeff = 0.01
        self.entropy_coeff_schedule = None
        self.tau = 1.0
        self.exploration_config = {'type': 'StochasticSampling'}

    @override(ImpalaConfig)
    def training(self, *, vtrace: Optional[bool]=NotProvided, use_critic: Optional[bool]=NotProvided, use_gae: Optional[bool]=NotProvided, lambda_: Optional[float]=NotProvided, clip_param: Optional[float]=NotProvided, use_kl_loss: Optional[bool]=NotProvided, kl_coeff: Optional[float]=NotProvided, kl_target: Optional[float]=NotProvided, tau: Optional[float]=NotProvided, target_update_frequency: Optional[int]=NotProvided, **kwargs) -> 'APPOConfig':
        """Sets the training related configuration.

        Args:
            vtrace: Whether to use V-trace weighted advantages. If false, PPO GAE
                advantages will be used instead.
            use_critic: Should use a critic as a baseline (otherwise don't use value
                baseline; required for using GAE). Only applies if vtrace=False.
            use_gae: If true, use the Generalized Advantage Estimator (GAE)
                with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
                Only applies if vtrace=False.
            lambda_: GAE (lambda) parameter.
            clip_param: PPO surrogate slipping parameter.
            use_kl_loss: Whether to use the KL-term in the loss function.
            kl_coeff: Coefficient for weighting the KL-loss term.
            kl_target: Target term for the KL-term to reach (via adjusting the
                `kl_coeff` automatically).
            tau: The factor by which to update the target policy network towards
                the current policy network. Can range between 0 and 1.
                e.g. updated_param = tau * current_param + (1 - tau) * target_param
            target_update_frequency: The frequency to update the target policy and
                tune the kl loss coefficients that are used during training. After
                setting this parameter, the algorithm waits for at least
                `target_update_frequency * minibatch_size * num_sgd_iter` number of
                samples to be trained on by the learner group before updating the target
                networks and tuned the kl loss coefficients that are used during
                training.
                NOTE: This parameter is only applicable when using the Learner API
                (_enable_new_api_stack=True).


        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if vtrace is not NotProvided:
            self.vtrace = vtrace
        if use_critic is not NotProvided:
            self.use_critic = use_critic
        if use_gae is not NotProvided:
            self.use_gae = use_gae
        if lambda_ is not NotProvided:
            self.lambda_ = lambda_
        if clip_param is not NotProvided:
            self.clip_param = clip_param
        if use_kl_loss is not NotProvided:
            self.use_kl_loss = use_kl_loss
        if kl_coeff is not NotProvided:
            self.kl_coeff = kl_coeff
        if kl_target is not NotProvided:
            self.kl_target = kl_target
        if tau is not NotProvided:
            self.tau = tau
        if target_update_frequency is not NotProvided:
            self.target_update_frequency = target_update_frequency
        return self

    @override(ImpalaConfig)
    def get_default_learner_class(self):
        if self.framework_str == 'torch':
            from ray.rllib.algorithms.appo.torch.appo_torch_learner import APPOTorchLearner
            return APPOTorchLearner
        elif self.framework_str == 'tf2':
            from ray.rllib.algorithms.appo.tf.appo_tf_learner import APPOTfLearner
            return APPOTfLearner
        else:
            raise ValueError(f"The framework {self.framework_str} is not supported. Use either 'torch' or 'tf2'.")

    @override(ImpalaConfig)
    def get_default_rl_module_spec(self) -> SingleAgentRLModuleSpec:
        if self.framework_str == 'torch':
            from ray.rllib.algorithms.appo.torch.appo_torch_rl_module import APPOTorchRLModule as RLModule
        elif self.framework_str == 'tf2':
            from ray.rllib.algorithms.appo.tf.appo_tf_rl_module import APPOTfRLModule as RLModule
        else:
            raise ValueError(f"The framework {self.framework_str} is not supported. Use either 'torch' or 'tf2'.")
        from ray.rllib.algorithms.appo.appo_catalog import APPOCatalog
        return SingleAgentRLModuleSpec(module_class=RLModule, catalog_class=APPOCatalog)

    @override(ImpalaConfig)
    def get_learner_hyperparameters(self) -> AppoLearnerHyperparameters:
        base_hps = super().get_learner_hyperparameters()
        return AppoLearnerHyperparameters(use_kl_loss=self.use_kl_loss, kl_target=self.kl_target, kl_coeff=self.kl_coeff, clip_param=self.clip_param, tau=self.tau, target_update_frequency_ts=self.train_batch_size * self.num_sgd_iter * self.target_update_frequency, **dataclasses.asdict(base_hps))