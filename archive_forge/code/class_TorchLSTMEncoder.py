from typing import Optional
import tree
from ray.rllib.core.models.base import (
from ray.rllib.core.models.base import Model, tokenize
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.primitives import TorchMLP, TorchCNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
class TorchLSTMEncoder(TorchModel, Encoder):
    """A recurrent LSTM encoder.

    This encoder has...
    - Zero or one tokenizers.
    - One or more LSTM layers.
    - One linear output layer.
    """

    def __init__(self, config: RecurrentEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        if config.tokenizer_config is not None:
            self.tokenizer = config.tokenizer_config.build(framework='torch')
            lstm_input_dims = config.tokenizer_config.output_dims
        else:
            self.tokenizer = None
            lstm_input_dims = config.input_dims
        assert len(lstm_input_dims) == 1
        lstm_input_dim = lstm_input_dims[0]
        self.lstm = nn.LSTM(lstm_input_dim, config.hidden_dim, config.num_layers, batch_first=config.batch_major, bias=config.use_bias)
        self._state_in_out_spec = {'h': TensorSpec('b, l, d', d=self.config.hidden_dim, l=self.config.num_layers, framework='torch'), 'c': TensorSpec('b, l, d', d=self.config.hidden_dim, l=self.config.num_layers, framework='torch')}

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return SpecDict({SampleBatch.OBS: TensorSpec('b, t, d', d=self.config.input_dims[0], framework='torch'), STATE_IN: self._state_in_out_spec})

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return SpecDict({ENCODER_OUT: TensorSpec('b, t, d', d=self.config.output_dims[0], framework='torch'), STATE_OUT: self._state_in_out_spec})

    @override(Model)
    def get_initial_state(self):
        return {'h': torch.zeros(self.config.num_layers, self.config.hidden_dim), 'c': torch.zeros(self.config.num_layers, self.config.hidden_dim)}

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        outputs = {}
        if self.tokenizer is not None:
            out = tokenize(self.tokenizer, inputs, framework='torch')
        else:
            out = inputs[SampleBatch.OBS].float()
        states_in = tree.map_structure(lambda s: s.transpose(0, 1), inputs[STATE_IN])
        out, states_out = self.lstm(out, (states_in['h'], states_in['c']))
        states_out = {'h': states_out[0], 'c': states_out[1]}
        outputs[ENCODER_OUT] = out
        outputs[STATE_OUT] = tree.map_structure(lambda s: s.transpose(0, 1), states_out)
        return outputs