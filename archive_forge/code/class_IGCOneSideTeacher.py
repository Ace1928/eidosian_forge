import csv
import os
from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Dict, Any
from parlai.core.build_data import download_multiprocess
from parlai.core.params import Opt
from parlai.core.teachers import AbstractImageTeacher
import parlai.utils.typing as PT
class IGCOneSideTeacher(ABC, IGCTeacher):
    """
    Override to only return one side of the conversation.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        super().add_cmdline_args(argparser)
        agent = argparser.add_argument_group('IGCResponseOnly Arguments')
        agent.add_argument('--igc-multi-ref', type='bool', default=False, help='specify true to evaluate on multi-reference labels')

    def num_episodes(self) -> int:
        return len(self.data)

    def num_examples(self) -> int:
        return len(self.data)

    @abstractmethod
    def get_label_key(self) -> str:
        """
        Return key into data dictionary for the label.
        """
        pass

    @abstractmethod
    def get_text(self, data) -> str:
        """
        Return text for an example.
        """
        pass

    def get(self, episode_idx: int, entry_idx: int=0) -> Dict[str, Any]:
        """
        Override to handle one-sided conversation.
        """
        data = self.data[episode_idx]
        image_id = data[self.image_id_key]
        if data[self.image_id_key] not in self.valid_image_ids:
            data[self.image_id_key] = self.blank_image_id
        image_features = self.get_image_features(data)
        labels = [data[self.get_label_key()]]
        if self.multi_ref:
            labels = data[f'multiref_{self.get_label_key()}s'].split('***')
        text = self.get_text(data)
        action = {'text': text, 'image_id': image_id, 'episode_done': True, 'image': image_features, 'labels': labels}
        return action