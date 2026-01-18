import logging
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import (
from transformers.file_utils import add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
import onnxruntime as ort
from ..exporters import TasksManager
from ..exporters.onnx import main_export
from ..modeling_base import FROM_PRETRAINED_START_DOCSTRING, OptimizedModel
from ..onnx.utils import _get_external_data_paths
from ..utils.file_utils import find_files_matching_pattern
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .io_binding import IOBindingHelper, TypeHelper
from .utils import (
@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForAudioClassification(ORTModel):
    """
    ONNX Model for audio-classification, with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting. This class officially supports audio_spectrogram_transformer, data2vec_audio, hubert, sew, sew_d, unispeech, unispeech_sat, wavlm, wav2vec2, wav2vec2-conformer.
    """
    auto_model_class = AutoModelForAudioClassification

    @add_start_docstrings_to_model_forward(ONNX_AUDIO_INPUTS_DOCSTRING.format('batch_size, sequence_length') + AUDIO_CLASSIFICATION_EXAMPLE.format(processor_class=_FEATURE_EXTRACTOR_FOR_DOC, model_class='ORTModelForAudioClassification', checkpoint='optimum/hubert-base-superb-ks'))
    def forward(self, input_values: Optional[torch.Tensor]=None, attenton_mask: Optional[torch.Tensor]=None, **kwargs):
        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        if self.device.type == 'cuda' and self.use_io_binding:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(input_values, ordered_input_names=self._ordered_input_names)
            io_binding.synchronize_inputs()
            self.model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            return SequenceClassifierOutput(logits=output_buffers['logits'].view(output_shapes['logits']))
        else:
            if use_torch:
                onnx_inputs = {'input_values': input_values.cpu().detach().numpy()}
            else:
                onnx_inputs = {'input_values': input_values}
            outputs = self.model.run(None, onnx_inputs)
            logits = outputs[self.output_names['logits']]
            if use_torch:
                logits = torch.from_numpy(logits).to(self.device)
            return SequenceClassifierOutput(logits=logits)