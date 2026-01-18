from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import TensorType, logging
class LongformerOnnxConfig(OnnxConfig):

    def __init__(self, config: 'PretrainedConfig', task: str='default', patching_specs: 'List[PatchingSpec]'=None):
        super().__init__(config, task, patching_specs)
        config.onnx_export = True

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == 'multiple-choice':
            dynamic_axis = {0: 'batch', 1: 'choice', 2: 'sequence'}
        else:
            dynamic_axis = {0: 'batch', 1: 'sequence'}
        return OrderedDict([('input_ids', dynamic_axis), ('attention_mask', dynamic_axis), ('global_attention_mask', dynamic_axis)])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        outputs = super().outputs
        if self.task == 'default':
            outputs['pooler_output'] = {0: 'batch'}
        return outputs

    @property
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        return 0.0001

    @property
    def default_onnx_opset(self) -> int:
        return max(super().default_onnx_opset, 14)

    def generate_dummy_inputs(self, tokenizer: 'PreTrainedTokenizerBase', batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None) -> Mapping[str, Any]:
        inputs = super().generate_dummy_inputs(preprocessor=tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework)
        import torch
        inputs['global_attention_mask'] = torch.zeros_like(inputs['input_ids'])
        inputs['global_attention_mask'][:, ::2] = 1
        return inputs