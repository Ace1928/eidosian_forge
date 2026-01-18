import abc
from typing import List, Optional, Tuple, Union
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.rnn_sequencing import get_fold_unfold_fns
from ray.rllib.utils.annotations import ExperimentalAPI, DeveloperAPI
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
@ExperimentalAPI
class ActorCriticEncoder(Encoder):
    """An encoder that potentially holds two stateless encoders.

    This is a special case of Encoder that can either enclose a single,
    shared encoder or two separate encoders: One for the actor and one for the
    critic. The two encoders are of the same type, and we can therefore make the
    assumption that they have the same input and output specs.
    """
    framework = None

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.shared:
            self.encoder = config.base_encoder_config.build(framework=self.framework)
        else:
            self.actor_encoder = config.base_encoder_config.build(framework=self.framework)
            self.critic_encoder = config.base_encoder_config.build(framework=self.framework)

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return [SampleBatch.OBS]

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return [(ENCODER_OUT, ACTOR), (ENCODER_OUT, CRITIC)]

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        if self.config.shared:
            encoder_outs = self.encoder(inputs, **kwargs)
            return {ENCODER_OUT: {ACTOR: encoder_outs[ENCODER_OUT], CRITIC: encoder_outs[ENCODER_OUT]}}
        else:
            actor_out = self.actor_encoder(inputs, **kwargs)
            critic_out = self.critic_encoder(inputs, **kwargs)
            return {ENCODER_OUT: {ACTOR: actor_out[ENCODER_OUT], CRITIC: critic_out[ENCODER_OUT]}}