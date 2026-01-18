from typing import Any, Dict, Optional, Tuple
import torch
from parlai.core.opt import Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .biencoder import AddLabelFixedCandsTRA
from .modules import (
from .transformer import TransformerRankerAgent
class IRFriendlyPolyencoderAgent(AddLabelFixedCandsTRA, PolyencoderAgent):
    """
    Poly-encoder agent that allows for adding label to fixed cands.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add cmd line args.
        """
        AddLabelFixedCandsTRA.add_cmdline_args(argparser)
        PolyencoderAgent.add_cmdline_args(argparser)