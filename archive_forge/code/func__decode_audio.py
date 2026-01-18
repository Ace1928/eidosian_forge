from typing import List, Optional
import numpy as np
from ...processing_utils import ProcessorMixin
from ...utils import to_numpy
def _decode_audio(self, audio_values, attention_mask: Optional=None) -> List[np.ndarray]:
    """
        This method strips any padding from the audio values to return a list of numpy audio arrays.
        """
    audio_values = to_numpy(audio_values)
    bsz, channels, seq_len = audio_values.shape
    if attention_mask is None:
        return list(audio_values)
    attention_mask = to_numpy(attention_mask)
    difference = seq_len - attention_mask.shape[-1]
    padding_value = 1 - self.feature_extractor.padding_value
    attention_mask = np.pad(attention_mask, ((0, 0), (0, difference)), 'constant', constant_values=padding_value)
    audio_values = audio_values.tolist()
    for i in range(bsz):
        sliced_audio = np.asarray(audio_values[i])[attention_mask[i][None, :] != self.feature_extractor.padding_value]
        audio_values[i] = sliced_audio.reshape(channels, -1)
    return audio_values