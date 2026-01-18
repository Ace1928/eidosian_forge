import re
from torch.nn import Module
from ..model import wav2vec2_model, Wav2Vec2Model
def _convert_state_dict(state_dict):
    converted = {}
    for k, v in state_dict.items():
        k = _map_key(k)
        if k is not None:
            converted[k] = v
    return converted