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
def _setup_topics_to_episodes(self) -> Dict[str, List[int]]:
    """
        Create a map from WoW topics to the indices of the WoW episodes that use them.
        """
    print('Starting to map topics to episodes.')
    topics_to_episodes = defaultdict(list)
    for episode_idx in range(self.wow_teacher.num_episodes()):
        topic = self.wow_teacher.get(episode_idx, entry_idx=0)['chosen_topic']
        topics_to_episodes[topic].append(episode_idx)
    print('Finished mapping topics to episodes.')
    return topics_to_episodes