import json
import os
from typing import Tuple
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from parlai.utils.typing import TShared
from .build import build
class ImageChatTestTeacher(ImageChatTeacher):
    """
    Test ImageChat teacher for ensuring pretrained model does not break.
    """

    def _setup_data(self, data_path, personalities_data_path):
        super()._setup_data(data_path, personalities_data_path)
        from parlai.zoo.image_chat.transresnet_multimodal import download
        download(self.opt['datapath'])
        image_features_path = os.path.join(self.opt['datapath'], 'models/image_chat/transresnet_multimodal/test_image_feats')
        import torch
        self.image_features = torch.load(image_features_path)

    def reset(self):
        """
        Reset teacher.
        """
        super().reset()
        self.example = None

    def num_episodes(self):
        """
        Return number of episodes.
        """
        return len(self.image_features)

    def num_examples(self):
        """
        Return number of examples.
        """
        return len(self.image_features)

    def get(self, episode_idx, entry_idx=0):
        """
        Get an example.

        :param episode_idx:
            index of episode in self.data
        :param entry_idx:
            optional, which entry in the episode to get

        :return:
            an example
        """
        data = self.data[episode_idx]
        personality, text = data['dialog'][entry_idx]
        episode_done = entry_idx == len(data['dialog']) - 1
        action = {'text': personality if self.include_personality else '', 'image': self.image_features[data['image_hash']], 'episode_done': episode_done, 'labels': [text]}
        if 'candidates' in data:
            action['label_candidates'] = data['candidates'][entry_idx][self.num_cands]
        return action