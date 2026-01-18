from collections import OrderedDict
from typing import Any, List, Mapping, Optional
from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec
from ...utils import logging
class GPT2OnnxConfig(OnnxConfigWithPast):

    def __init__(self, config: PretrainedConfig, task: str='default', patching_specs: List[PatchingSpec]=None, use_past: bool=False):
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        if not getattr(self._config, 'pad_token_id', None):
            self._config.pad_token_id = 0

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({'input_ids': {0: 'batch', 1: 'sequence'}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction='inputs')
            common_inputs['attention_mask'] = {0: 'batch', 1: 'past_sequence + sequence'}
        else:
            common_inputs['attention_mask'] = {0: 'batch', 1: 'sequence'}
        return common_inputs

    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    def generate_dummy_inputs(self, tokenizer: PreTrainedTokenizer, batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None) -> Mapping[str, Any]:
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework)
        ordered_inputs = OrderedDict({'input_ids': common_inputs['input_ids']})
        if self.use_past:
            if not is_torch_available():
                raise ValueError('Cannot generate dummy past_keys inputs without PyTorch installed.')
            else:
                import torch
                batch, seqlen = common_inputs['input_ids'].shape
                past_key_values_length = seqlen + 2
                past_shape = (batch, self.num_attention_heads, past_key_values_length, self._config.hidden_size // self.num_attention_heads)
                ordered_inputs['past_key_values'] = [(torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)]
        ordered_inputs['attention_mask'] = common_inputs['attention_mask']
        if self.use_past:
            mask_dtype = ordered_inputs['attention_mask'].dtype
            ordered_inputs['attention_mask'] = torch.cat([ordered_inputs['attention_mask'], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1)
        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13