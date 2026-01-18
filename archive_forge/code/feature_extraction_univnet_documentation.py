from typing import Any, Dict, List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging

        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the input `raw_speech` waveforms (according to the model's padding side and
                padding index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).

                If `pad_end = True`, that padding will occur before the `padding` strategy is applied.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*, defaults to `True`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_noise (`bool`, *optional*, defaults to `True`):
                Whether to generate and return a noise waveform for use in [`UnivNetModel.forward`].
            generator (`numpy.random.Generator`, *optional*, defaults to `None`):
                An optional `numpy.random.Generator` random number generator to use when generating noise.
            pad_end (`bool`, *optional*, defaults to `False`):
                Whether to pad the end of each waveform with silence. This can help reduce artifacts at the end of the
                generated audio sample; see https://github.com/seungwonpark/melgan/issues/8 for more details. This
                padding will be done before the padding strategy specified in `padding` is performed.
            pad_length (`int`, *optional*, defaults to `None`):
                If padding the end of each waveform, the length of the padding in spectrogram frames. If not set, this
                will default to `self.config.pad_end_length`.
            do_normalize (`bool`, *optional*):
                Whether to perform Tacotron 2 normalization on the input. Normalizing can help to significantly improve
                the performance for some models. If not set, this will default to `self.config.do_normalize`.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.np.array` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        