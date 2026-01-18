import warnings
from typing import Any, Dict, List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
def _process_audio(self, speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]], is_target: bool=False, padding: Union[bool, str, PaddingStrategy]=False, max_length: Optional[int]=None, truncation: bool=False, pad_to_multiple_of: Optional[int]=None, return_attention_mask: Optional[bool]=None, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> BatchFeature:
    is_batched_numpy = isinstance(speech, np.ndarray) and len(speech.shape) > 1
    if is_batched_numpy and len(speech.shape) > 2:
        raise ValueError(f'Only mono-channel audio is supported for input to {self}')
    is_batched = is_batched_numpy or (isinstance(speech, (list, tuple)) and isinstance(speech[0], (np.ndarray, tuple, list)))
    if is_batched:
        speech = [np.asarray(speech, dtype=np.float32) for speech in speech]
    elif not is_batched and (not isinstance(speech, np.ndarray)):
        speech = np.asarray(speech, dtype=np.float32)
    elif isinstance(speech, np.ndarray) and speech.dtype is np.dtype(np.float64):
        speech = speech.astype(np.float32)
    if not is_batched:
        speech = [speech]
    feature_size_hack = self.feature_size
    if is_target:
        features = [self._extract_mel_features(waveform) for waveform in speech]
        encoded_inputs = BatchFeature({'input_values': features})
        self.feature_size = self.num_mel_bins
    else:
        encoded_inputs = BatchFeature({'input_values': speech})
    padded_inputs = self.pad(encoded_inputs, padding=padding, max_length=max_length, truncation=truncation, pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask, **kwargs)
    self.feature_size = feature_size_hack
    input_values = padded_inputs['input_values']
    if not isinstance(input_values[0], np.ndarray):
        padded_inputs['input_values'] = [np.asarray(array, dtype=np.float32) for array in input_values]
    elif not isinstance(input_values, np.ndarray) and isinstance(input_values[0], np.ndarray) and (input_values[0].dtype is np.dtype(np.float64)):
        padded_inputs['input_values'] = [array.astype(np.float32) for array in input_values]
    elif isinstance(input_values, np.ndarray) and input_values.dtype is np.dtype(np.float64):
        padded_inputs['input_values'] = input_values.astype(np.float32)
    attention_mask = padded_inputs.get('attention_mask')
    if attention_mask is not None:
        padded_inputs['attention_mask'] = [np.asarray(array, dtype=np.int32) for array in attention_mask]
    if not is_target and self.do_normalize:
        attention_mask = attention_mask if self._get_padding_strategies(padding, max_length=max_length) is not PaddingStrategy.DO_NOT_PAD else None
        padded_inputs['input_values'] = self.zero_mean_unit_var_norm(padded_inputs['input_values'], attention_mask=attention_mask, padding_value=self.padding_value)
    if return_tensors is not None:
        padded_inputs = padded_inputs.convert_to_tensors(return_tensors)
    return padded_inputs