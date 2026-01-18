import copy
from typing import Any, Dict, List, Optional, Union
import numpy as np
from ...audio_utils import chroma_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_torch_available, is_torchaudio_available, logging
def _extract_stem_indices(self, audio, sampling_rate=None):
    """
        Extracts stems from the output of the [Demucs](https://github.com/adefossez/demucs/tree/main) audio separation model,
        then converts to mono-channel and resample to the feature extractor sampling rate.

        Args:
            audio (`torch.Tensor` of shape `(batch_size, num_stems, channel_size, audio_length)`):
                The output of the Demucs model to be processed.
            sampling_rate (`int`, *optional*):
                Demucs sampling rate. If not specified, defaults to `44000`.
        """
    sampling_rate = 44000 if sampling_rate is None else sampling_rate
    wav = audio[:, torch.tensor(self.stem_indices)]
    wav = wav.sum(1)
    wav = wav.mean(dim=1, keepdim=True)
    if sampling_rate != self.sampling_rate:
        wav = torchaudio.functional.resample(wav, sampling_rate, self.sampling_rate, rolloff=0.945, lowpass_filter_width=24)
    wav = wav.squeeze(1)
    return wav