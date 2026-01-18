from copy import deepcopy
import json
import random
import os
import string
from parlai.core.agents import create_agent
from parlai.core.message import Message
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
from parlai.utils.misc import warn_once
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
class InteractiveGeneratorWorld(InteractiveWorld):
    """
    Interactive world for generative models.

    Specifically a world for models trained on the task `-t wizard_of_wikipedia
    generator`.
    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('[ loading topics.. ]')
        self.opt = opt
        self._load_topics(opt)
        self.num_topics = opt['num_topics']
        self.cnt = 0
        self.human_agent = self.agents[0]
        self.model_agent = self.agents[1]
        self._set_up_knowledge_agent(add_token_knowledge=True)

    def _add_knowledge_to_act(self, act):
        act = super()._add_knowledge_to_act(act)
        if self.opt.get('prepend_gold_knowledge', False):
            warn_once('Prepending selected knowledge to dialogue input.If this was not intended behavior, please run with the flag --prepend-gold-knowledge False')
            knowledge_text = ' '.join([TOKEN_KNOWLEDGE, act['checked_sentence'], TOKEN_END_KNOWLEDGE])
            new_text = '\n'.join([knowledge_text, act['text']])
            act.force_set('text', new_text)
        else:
            warn_once('Not prepending selected knowledge to dialogue input.If this was not intended behavior, please run with the flag --prepend-gold-knowledge True')
        return act