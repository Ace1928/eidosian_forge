import torch
from .transformer import TransformerRankerAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
def _vectorize_text(self, *args, **kwargs):
    """
        Override to add start end tokens.

        necessary for fixed cands.
        """
    if 'add_start' in kwargs:
        kwargs['add_start'] = True
        kwargs['add_end'] = True
    return super()._vectorize_text(*args, **kwargs)