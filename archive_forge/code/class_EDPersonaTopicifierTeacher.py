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
class EDPersonaTopicifierTeacher(EmpatheticDialoguesTeacher):
    """
    Adds persona and WoW topic to ED context strings.
    """
    RECOMPILE_DEFAULT = False

    @classmethod
    def add_cmdline_args(cls, argparser):
        EmpatheticDialoguesTeacher.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('EDPersonaTopicifierTeacher arguments')
        agent.add_argument('--recompile-persona-topic-data', type='bool', default=cls.RECOMPILE_DEFAULT, help='Re-compile data with ConvAI2 personas and WoW topics added. Only useful for demonstrating how data was produced.')

    def __init__(self, opt, shared=None):
        self.persona_topicifier = PersonaTopicifier(opt=opt, should_have_personas=False, should_have_topics=False)
        super().__init__(opt, shared=shared)
        if self.remove_political_convos is True or self.opt.get('deepmoji') is not None or self.opt.get('fasttextloc') is not None or (self.opt.get('prepend', -1) > 0):
            raise NotImplementedError('Removing political conversations or using deepmoji, fasttextloc, or prepend not supported with this teacher.')
        if opt.get('recompile_persona_topic_data', self.RECOMPILE_DEFAULT):
            self.data_path = _cached_data_path(opt=self.opt, experiencer_side_only=self.experiencer_side_only) + '.recompiled'
            warn_once(f'Compiling data file for {self.data_path}.')
            self.persona_topic_data = self._compile_data()
            warn_once(f'Saving data to {self.data_path}.')
            with open(self.data_path, 'w') as f_write:
                json.dump(self.persona_topic_data, f_write)
        else:
            self.data_path = _cached_data_path(opt=self.opt, experiencer_side_only=self.experiencer_side_only)
            warn_once(f'Loading cached data from {self.data_path}.')
            with open(self.data_path, 'r') as f_read:
                self.persona_topic_data = json.load(f_read)

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

    def _get_example(self, episode_idx: int, entry_idx: Optional[int]=None):
        """
        Get example from the base ED teacher and add persona and WoW topic strings.
        """
        gotten = super().get(episode_idx, entry_idx=entry_idx)
        if entry_idx == 0:
            modified_text = self.persona_topicifier.get_modified_text(gotten['text'])
            gotten['text'] = modified_text
        return gotten

    def get(self, episode_idx: int, entry_idx: Optional[int]=None) -> dict:
        """
        Get example from the final data with personas and WoW topic strings.
        """
        if entry_idx is None:
            entry_idx = 0
        return self.persona_topic_data[episode_idx][entry_idx]