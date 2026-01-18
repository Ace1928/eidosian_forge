from typing import List, Optional
import torch
from torch.backends._nnapi.serializer import _NnapiSerializer
class NnapiInterfaceWrapper(torch.nn.Module):
    """NNAPI list-ifying and de-list-ifying wrapper.

        NNAPI always expects a list of inputs and provides a list of outputs.
        This module allows us to accept inputs as separate arguments.
        It returns results as either a single tensor or tuple,
        matching the original module.
        """

    def __init__(self, mod):
        super().__init__()
        self.mod = mod