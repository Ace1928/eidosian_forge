from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import is_tf_available
from ...onnx import merge_decoders
from ...utils import (
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import DecoderModelPatcher
class TextDecoderOnnxConfig(OnnxConfigWithPast):
    """
    Handles decoder-based text architectures.
    """
    PAD_ATTENTION_MASK_TO_PAST = True
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator

    def __init__(self, config: 'PretrainedConfig', task: str='feature-extraction', int_dtype: str='int64', float_dtype: str='fp32', use_past: bool=False, use_past_in_inputs: bool=False, preprocessors: Optional[List[Any]]=None, legacy: bool=False):
        super().__init__(config=config, task=task, int_dtype=int_dtype, float_dtype=float_dtype, use_past=use_past, use_past_in_inputs=use_past_in_inputs, preprocessors=preprocessors, legacy=legacy)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.use_past_in_inputs:
            common_inputs = {'input_ids': {0: 'batch_size', 1: 'sequence_length'}}
            self.add_past_key_values(common_inputs, direction='inputs')
            common_inputs['attention_mask'] = {0: 'batch_size', 1: 'past_sequence_length + 1'}
        else:
            common_inputs = {'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 'attention_mask': {0: 'batch_size', 1: 'sequence_length'}}
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self.is_merged is False:
            common_outputs = super().outputs
        else:
            common_outputs = OrderedDict({'logits': {0: 'batch_size', 1: 'sequence_length'}})
            self.add_past_key_values(common_outputs, direction='outputs')
        return common_outputs

    def post_process_exported_models(self, path: Path, models_and_onnx_configs: Dict[str, Tuple[Union['PreTrainedModel', 'TFPreTrainedModel', 'ModelMixin'], 'OnnxConfig']], onnx_files_subpaths: List[str]):
        models_and_onnx_configs, onnx_files_subpaths = super().post_process_exported_models(path, models_and_onnx_configs, onnx_files_subpaths)
        if self.use_past is True and len(models_and_onnx_configs) == 2:
            decoder_path = Path(path, onnx_files_subpaths[0])
            decoder_with_past_path = Path(path, onnx_files_subpaths[1])
            decoder_merged_path = Path(path, ONNX_DECODER_MERGED_NAME + '.onnx')
            try:
                merge_decoders(decoder=decoder_path, decoder_with_past=decoder_with_past_path, save_path=decoder_merged_path)
            except Exception as e:
                raise Exception(f'Unable to merge decoders. Detailed error: {e}')
            onnx_files_subpaths = [decoder_merged_path.name, decoder_merged_path.name]
            models_and_onnx_configs[ONNX_DECODER_NAME][1].is_merged = True
            models_and_onnx_configs[ONNX_DECODER_NAME][1].use_cache_branch = False
            models_and_onnx_configs[ONNX_DECODER_NAME][1].use_past_in_inputs = True
            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1].use_cache_branch = True
            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1].is_merged = True
        return (models_and_onnx_configs, onnx_files_subpaths)

    def patch_model_for_export(self, model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Optional[Dict[str, Any]]=None) -> 'ModelPatcher':
        return DecoderModelPatcher(self, model, model_kwargs=model_kwargs)