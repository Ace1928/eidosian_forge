import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def _build_personality_dictionary(self, personalities_list):
    """
        Build the personality dictionary mapping personality to id.

        :param personalities_list:
            list of personalities
        """
    self.personalities_list = personalities_list
    self.personality_to_id = {p: i for i, p in enumerate(personalities_list)}
    self.num_personalities = len(self.personalities_list) + 1