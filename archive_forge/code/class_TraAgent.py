from parlai.core.torch_ranker_agent import TorchRankerAgent
import torch
from torch import nn
class TraAgent(TorchRankerAgent):
    """
    Example subclass of TorchRankerAgent.

    This particular implementation is a simple bag-of-words model, which demonstrates
    the minimum implementation requirements to make a new ranking model.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add CLI args.
        """
        TorchRankerAgent.add_cmdline_args(argparser)
        arg_group = argparser.add_argument_group('ExampleBagOfWordsModel Arguments')
        arg_group.add_argument('--hidden-dim', type=int, default=512)

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        This function takes in a Batch object as well as a Tensor of candidate vectors.

        It must return a list of scores corresponding to the likelihood that the
        candidate vector at that index is the proper response. If `cand_encs` is not
        None (when we cache the encoding of the candidate vectors), you may use these
        instead of calling self.model on `cand_vecs`.
        """
        scores = self.model.forward(batch, cand_vecs, cand_encs)
        return scores

    def build_model(self):
        """
        This function is required to build the model and assign to the object
        `self.model`.
        """
        return ExampleBagOfWordsModel(self.opt, self.dict)