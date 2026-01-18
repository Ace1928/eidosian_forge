import copy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging

        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            truncation (`str`, *optional*):
                Truncation pattern for long audio inputs. Two patterns are available:
                    - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and
                      a downsampled version of the entire mel spectrogram.
                If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a
                copy of the original mel obtained from the padded audio.
                    - `rand_trunc` will select a random crop of the mel spectrogram.
            padding (`str`, *optional*):
               Padding pattern for shorter audio inputs. Three patterns were originally implemented:
                    - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
                    - `repeat`: the audio is repeated and then cut to fit the `max_length`
                    - `pad`: the audio is padded.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.np.array` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
        