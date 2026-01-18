import torch
from .transformer import TransformerRankerAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
class IRFriendlyBiencoderAgent(AddLabelFixedCandsTRA, BiencoderAgent):
    """
    Bi-encoder agent that allows for adding label to fixed cands.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add cmd line args.
        """
        AddLabelFixedCandsTRA.add_cmdline_args(argparser)
        BiencoderAgent.add_cmdline_args(argparser)