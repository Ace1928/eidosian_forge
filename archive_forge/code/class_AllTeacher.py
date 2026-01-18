import copy
import json
import os
import random
import re
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from parlai.core.opt import Opt
from parlai.core.teachers import (
from parlai.tasks.convai2.agents import (
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import WizardDialogKnowledgeTeacher
from parlai.utils.misc import warn_once
from .build import build
class AllTeacher(MultiTaskTeacher):
    """
    Multitask teacher that combines all "Persona Topicifier" teachers.
    """

    def __init__(self, opt, shared=None):
        topicifier_tasks = ['blended_skill_talk:ConvAI2PersonaTopicifier', 'blended_skill_talk:EDPersonaTopicifier', 'blended_skill_talk:WoWPersonaTopicifier', 'blended_skill_talk:BlendedSkillTalk']
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(topicifier_tasks)
        super().__init__(opt, shared)