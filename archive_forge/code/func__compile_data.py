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
def _compile_data(self) -> List[List[dict]]:
    """
        Compile data to be saved for faster future use.
        """
    warn_once(f'Starting to compile {self.num_episodes():d} episodes.')
    all_data = []
    for episode_idx in tqdm(range(self.num_episodes())):
        episode_data = []
        entry_idx = 0
        while True:
            example_data = self._get_example(episode_idx=episode_idx, entry_idx=entry_idx)
            episode_data.append(example_data)
            if example_data['episode_done']:
                all_data.append(episode_data)
                break
            else:
                entry_idx += 1
    return all_data