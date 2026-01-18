import subprocess
from typing import Union
import numpy as np
import requests
from ..utils import add_end_docstrings, is_torch_available, is_torchaudio_available, logging
from .base import Pipeline, build_pipeline_init_args

        Classify the sequence(s) given as inputs. See the [`AutomaticSpeechRecognitionPipeline`] documentation for more
        information.

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                The inputs is either :
                    - `str` that is the filename of the audio file, the file will be read at the correct sampling rate
                      to get the waveform using *ffmpeg*. This requires *ffmpeg* to be installed on the system.
                    - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
                      same way.
                    - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                        Raw audio at the correct sampling rate (no further check will be done)
                    - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
                      pipeline do the resampling. The dict must be either be in the format `{"sampling_rate": int,
                      "raw": np.array}`, or `{"sampling_rate": int, "array": np.array}`, where the key `"raw"` or
                      `"array"` is used to denote the raw audio waveform.
            top_k (`int`, *optional*, defaults to None):
                The number of top labels that will be returned by the pipeline. If the provided number is `None` or
                higher than the number of labels available in the model configuration, it will default to the number of
                labels.

        Return:
            A list of `dict` with the following keys:

            - **label** (`str`) -- The label predicted.
            - **score** (`float`) -- The corresponding probability.
        