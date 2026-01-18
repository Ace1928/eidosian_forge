from parlai.core.agents import create_agent_from_shared
from parlai.core.message import Message
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
import parlai.mturk.core.mturk_utils as mutils
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.tasks.wizard_of_wikipedia.build import build
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
from joblib import Parallel, delayed
import json
import numpy as np
import os
import pickle
import random
import time
def _add_knowledge_to_act(self, act):
    self.knowledge_agent.observe(act, actor_id='apprentice')
    knowledge_act = self.knowledge_agent.act()
    act['knowledge'] = knowledge_act['text']
    act['checked_sentence'] = knowledge_act['checked_sentence']
    print('[ Using chosen sentence from Wikpedia ]: {}'.format(knowledge_act['checked_sentence']))
    act['title'] = knowledge_act['title']
    if self.opt.get('prepend_gold_knowledge', False):
        knowledge_text = ' '.join([TOKEN_KNOWLEDGE, knowledge_act['checked_sentence'], TOKEN_END_KNOWLEDGE])
        new_text = '\n'.join([knowledge_text, act['text']])
        if isinstance(act, Message):
            act.force_set('text', new_text)
        else:
            act['text'] = new_text
    return act