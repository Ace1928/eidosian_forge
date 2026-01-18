from typing import Type, Optional
from ray.rllib.algorithms.sac import (
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.sac.rnnsac_torch_policy import RNNSACTorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
class RNNSACConfig(SACConfig):
    """Defines a configuration class from which an RNNSAC can be built.

    Example:
        >>> config = RNNSACConfig().training(gamma=0.9, lr=0.01)        ...     .resources(num_gpus=0)        ...     .rollouts(num_rollout_workers=4)
        >>> print(config.to_dict())  # doctest: +SKIP
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")
        >>> algo.train()  # doctest: +SKIP
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or RNNSAC)
        self.framework_str = 'torch'
        self.batch_mode = 'complete_episodes'
        self.zero_init_states = True
        self.replay_buffer_config = {'storage_unit': 'sequences', 'replay_burn_in': 0, 'replay_sequence_length': -1}
        self.burn_in = DEPRECATED_VALUE

    @override(SACConfig)
    def training(self, *, zero_init_states: Optional[bool]=NotProvided, **kwargs) -> 'RNNSACConfig':
        """Sets the training related configuration.

        Args:
            zero_init_states: If True, assume a zero-initialized state input (no matter
                where in the episode the sequence is located).
                If False, store the initial states along with each SampleBatch, use
                it (as initial state when running through the network for training),
                and update that initial state during training (from the internal
                state outputs of the immediately preceding sequence).

        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if zero_init_states is not NotProvided:
            self.zero_init_states = zero_init_states
        return self

    @override(SACConfig)
    def validate(self) -> None:
        super().validate()
        replay_sequence_length = self.replay_buffer_config['replay_burn_in'] + self.model['max_seq_len']
        if self.replay_buffer_config.get('replay_sequence_length', None) not in [None, -1, replay_sequence_length]:
            raise ValueError("`replay_sequence_length` is calculated automatically to be config.model['max_seq_len'] + config['replay_burn_in']. Leave config.replay_buffer_config['replay_sequence_length'] blank to avoid this error.")
        self.replay_buffer_config['replay_sequence_length'] = replay_sequence_length
        if self.framework_str != 'torch':
            raise ValueError('Only `framework=torch` supported so far for RNNSAC!')