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
class StatefulActorCriticEncoder(Encoder):
    """An encoder that potentially holds two potentially stateful encoders.

    This is a special case of Encoder that can either enclose a single,
    shared encoder or two separate encoders: One for the actor and one for the
    critic. The two encoders are of the same type, and we can therefore make the
    assumption that they have the same input and output specs.

    If this encoder wraps a single encoder, state in input- and output dicts
    is simply stored under the key `STATE_IN` and `STATE_OUT`, respectively.
    If this encoder wraps two encoders, state in input- and output dicts is
    stored under the keys `(STATE_IN, ACTOR)` and `(STATE_IN, CRITIC)` and
    `(STATE_OUT, ACTOR)` and `(STATE_OUT, CRITIC)`, respectively.
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
        return [SampleBatch.OBS, STATE_IN]

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return [(ENCODER_OUT, ACTOR), (ENCODER_OUT, CRITIC), (STATE_OUT,)]

    @override(Model)
    def get_initial_state(self):
        if self.config.shared:
            return self.encoder.get_initial_state()
        else:
            return {ACTOR: self.actor_encoder.get_initial_state(), CRITIC: self.critic_encoder.get_initial_state()}

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        outputs = {}
        if self.config.shared:
            outs = self.encoder(inputs, **kwargs)
            encoder_out = outs.pop(ENCODER_OUT)
            outputs[ENCODER_OUT] = {ACTOR: encoder_out, CRITIC: encoder_out}
            outputs[STATE_OUT] = outs[STATE_OUT]
        else:
            actor_inputs = inputs.copy()
            critic_inputs = inputs.copy()
            actor_inputs[STATE_IN] = inputs[STATE_IN][ACTOR]
            critic_inputs[STATE_IN] = inputs[STATE_IN][CRITIC]
            actor_out = self.actor_encoder(actor_inputs, **kwargs)
            critic_out = self.critic_encoder(critic_inputs, **kwargs)
            outputs[ENCODER_OUT] = {ACTOR: actor_out[ENCODER_OUT], CRITIC: critic_out[ENCODER_OUT]}
            outputs[STATE_OUT] = {ACTOR: actor_out[STATE_OUT], CRITIC: critic_out[STATE_OUT]}
        return outputs