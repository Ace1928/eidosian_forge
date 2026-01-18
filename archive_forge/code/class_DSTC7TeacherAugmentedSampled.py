from parlai.core.teachers import FixedDialogTeacher
from .build import build
import json
import os
import random
class DSTC7TeacherAugmentedSampled(DSTC7Teacher):
    """
    The dev and test set are the same, but the training set has been augmented using the
    other utterances.

    Moreover, only 16 candidates are used (including the right one)
    """

    def get_suffix(self):
        if self.split != 'train':
            return ''
        return '_sampled'

    def get_nb_cands(self):
        return 16

    def get(self, episode_idx, entry_idx=0):
        sample = super().get(episode_idx, entry_idx)
        if self.split != 'train':
            return sample
        new_cands = [sample['labels'][0]]
        counter = 0
        while len(new_cands) < self.get_nb_cands():
            if sample['label_candidates'][counter] not in sample['labels']:
                new_cands.append(sample['label_candidates'][counter])
            counter += 1
        sample['label_candidates'] = new_cands
        return sample