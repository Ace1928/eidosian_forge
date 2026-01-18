from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import recursive_getattr
from parlai.utils.logging import logging
from .modules import (
import torch
class TransformerGeneratorAgent(TorchGeneratorAgent):
    """
    TransformerGeneratorAgent.

    Implementation of TorchGeneratorAgent, where the model is a Transformer
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)
        super(TransformerGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(model.encoder.embeddings.weight, self.opt['embedding_type'])
        return model

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Resize the token embeddings when are adding extra special tokens.
        """
        new_size = self.model.embeddings.weight.size()[0]
        orig_size = state_dict['embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {orig_size} to {new_size}')
        if new_size <= orig_size:
            raise RuntimeError(msg)
        for emb_weights in ['embeddings.weight', 'encoder.embeddings.weight', 'decoder.embeddings.weight']:
            old_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.model, emb_weights).to(old_embs.device)
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            state_dict[emb_weights] = new_embs
        return state_dict