import re
from torch.nn import Module
from ..model import wav2vec2_model, Wav2Vec2Model
def _import_wav2vec2_finetuning(original: Module) -> Wav2Vec2Model:
    config = _parse_config(original.w2v_model)
    model = wav2vec2_model(**config, aux_num_out=original.proj.out_features)
    model.load_state_dict(_convert_state_dict(original.state_dict()))
    return model