import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import torch
from onnx.tools import update_model_dims
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast
import onnxruntime
from ..exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
from ..onnx.utils import check_model_uses_external_data
from ..utils import NormalizedConfigManager, check_if_transformers_greater
from ..utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ..utils.save_utils import maybe_save_preprocessors
from .constants import DECODER_MERGED_ONNX_FILE_PATTERN, DECODER_ONNX_FILE_PATTERN, DECODER_WITH_PAST_ONNX_FILE_PATTERN
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .models.bloom import bloom_convert_to_bloom_cache, bloom_convert_to_standard_cache
from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_WEIGHTS_NAME
@classmethod
def _from_transformers(cls, model_id: str, config: 'PretrainedConfig', use_auth_token: Optional[Union[bool, str]]=None, revision: str='main', force_download: bool=True, cache_dir: Optional[str]=None, subfolder: str='', local_files_only: bool=False, trust_remote_code: bool=False, use_cache: bool=True, use_merged: bool=False, provider: str='CPUExecutionProvider', session_options: Optional[onnxruntime.SessionOptions]=None, provider_options: Optional[Dict[str, Any]]=None, use_io_binding: Optional[bool]=None, task: Optional[str]=None) -> 'ORTModelForCausalLM':
    file_name = ONNX_WEIGHTS_NAME
    if use_merged:
        logger.warning('The `use_merged` argument is deprecated when the model is exported, and not used anymore.')
        use_merged = False
    if task is None:
        task = cls._auto_model_to_task(cls.auto_model_class)
        if use_cache:
            task += '-with-past'
    save_dir = TemporaryDirectory()
    save_dir_path = Path(save_dir.name)
    main_export(model_name_or_path=model_id, output=save_dir_path, task=task, do_validation=False, no_post_process=False, legacy=False, subfolder=subfolder, revision=revision, cache_dir=cache_dir, use_auth_token=use_auth_token, local_files_only=local_files_only, force_download=force_download, trust_remote_code=trust_remote_code)
    config.save_pretrained(save_dir_path)
    maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)
    return cls._from_pretrained(save_dir_path, config, use_cache=use_cache, use_merged=use_merged, provider=provider, session_options=session_options, provider_options=provider_options, use_io_binding=use_io_binding, model_save_dir=save_dir, file_name=file_name)