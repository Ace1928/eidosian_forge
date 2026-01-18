from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel
from parlai.agents.hugging_face.dict import Gpt2DictionaryAgent
from parlai.utils.misc import warn_once
from parlai.utils.torch import IdentityLayer, concat_without_padding, padded_tensor
import torch
def _get_special_tokens(self, opt, dict):
    return (dict.null_idx, dict.start_idx, dict.end_idx)