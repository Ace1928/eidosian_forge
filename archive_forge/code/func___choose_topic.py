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
def __choose_topic(self, persona):
    persona_lines = persona.strip().split('\n')
    for p in persona_lines:
        p_str = p.replace('your persona:', '')
        p_str = p_str.strip()
        if p_str in self.persona_strings_to_wow_topics_map:
            topics = self.persona_strings_to_wow_topics_map[p_str]
            topic = topics[0] + '\n'
            return topic
    for utt, topics in self.persona_strings_to_wow_topics_map.items():
        utt_words = utt.split()
        utt_words_long = [utt for utt in utt_words if len(utt) > 6]
        for long_utt in utt_words_long:
            if long_utt in persona:
                return topics[0] + '\n'
    return topics[0] + '\n'