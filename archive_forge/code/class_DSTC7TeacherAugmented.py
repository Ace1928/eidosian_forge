from parlai.core.teachers import FixedDialogTeacher
from .build import build
import json
import os
import random
class DSTC7TeacherAugmented(DSTC7Teacher):
    """
    Augmented Data.

    To mimic the way ParlAI generally handles dialogue datasets, the data associated
    with this teacher is presented in a format such that a single "episode" is split
    across multiple entries.

    I.e., suppose we have the following dialogue between speakers 1 and 2:
    utterances: [A, B, C, D, E],
    label: F

    The data in this file is split such that we have the following episodes:

    ep1:
        utterances: [A],
        label: B
    ep2:
        utterances [A, B, C]
        label: D
    ep3:
        utterances: [A, B, C, D, E],
        label: F
    """

    def get_suffix(self):
        if self.split != 'train':
            return ''
        return '_augmented'