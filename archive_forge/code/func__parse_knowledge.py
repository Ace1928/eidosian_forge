from itertools import chain
from functools import lru_cache
import torch as th
import numpy as np
from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from .modules import EndToEndModel
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE
def _parse_knowledge(self, obs):
    if 'knowledge_parsed' in obs:
        return list(obs['knowledge_parsed'])
    if 'checked_sentence' not in obs:
        obs_know = [k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')]
        obs_know = [k for k in obs_know if k]
        obs['knowledge_parsed'] = obs_know
        return obs['knowledge_parsed']
    checked_sentence = '{} {} {}'.format(obs['title'], TOKEN_KNOWLEDGE, obs['checked_sentence'])
    obs_know = [k.strip() for k in obs.get('knowledge', 'no_passages_used').split('\n')]
    obs_know = [k for k in obs_know if k]
    try:
        i = obs_know.index(checked_sentence)
    except ValueError:
        i = 0
        obs_know[0] = checked_sentence
    obs_know[0], obs_know[i] = (obs_know[i], obs_know[0])
    obs['knowledge_parsed'] = obs_know
    obs['checked_sentence_parsed'] = checked_sentence
    return obs['knowledge_parsed']