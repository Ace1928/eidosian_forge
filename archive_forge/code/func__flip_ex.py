from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.light_dialog.agents import DefaultTeacher as OrigLightTeacher
from parlai.tasks.light_genderation_bias.build import build
from collections import deque
from copy import deepcopy
import csv
import json
import os
import random
def _flip_ex(self, ex):
    """
        Return the counterfactual example for a given example (i.e. swap 'he' --> 'she')
        """
    new_ex = deepcopy(ex)
    text = ex['text']
    labels = 'labels' if 'labels' in ex else 'eval_labels'
    label = ex[labels][0]
    new_ex.force_set('text', self._flip_str(text))
    new_ex.force_set(labels, [self._flip_str(label)])
    return new_ex