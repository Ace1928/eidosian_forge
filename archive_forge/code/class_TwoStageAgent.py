from itertools import chain
from functools import lru_cache
import torch as th
import numpy as np
from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from .modules import EndToEndModel
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
class TwoStageAgent(_GenericWizardAgent):

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if shared is not None:
            self.dict[TOKEN_DIALOG] = 9999999

    def _set_text_vec(self, obs, history, truncate):
        if 'text' not in obs:
            return obs
        if 'text_vec' not in obs:
            fields = []
            dialogue_history = history.get_history_str()
            if 'chosen_topic' in obs:
                fields += [obs['title']]
            if 'checked_sentence' in obs:
                fields += [TOKEN_KNOWLEDGE, obs['checked_sentence']]
            if dialogue_history:
                fields += [TOKEN_DIALOG, dialogue_history]
            obs['text'] = ' '.join(fields)
            obs['text_vec'] = self.dict.txt2vec(obs['text'])
        if 'text_vec' in obs:
            obs['text_vec'] = th.LongTensor(self._check_truncate(obs['text_vec'], truncate, True))
        return obs