import json
import os
import re
import numpy as np
from collections import Counter
from parlai.core.agents import Agent
from collections import defaultdict
from parlai.core.teachers import FixedDialogTeacher
from parlai.core.image_featurizers import ImageLoader
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage as buildImage_2014
from parlai.tasks.coco_caption.build_2015 import buildImage as buildImage_2015
class McTeacher(OeTeacher):
    """
    VQA Multiple-Choice teacher, which inherits from OeTeacher but overrides the label
    and label_candidates fields with multiple choice data.
    """

    def get(self, episode_idx, entry_idx=0):
        action = super().get(episode_idx, entry_idx)
        qa = self.ques['questions'][episode_idx]
        multiple_choices = qa['multiple_choices']
        action['label_candidates'] = multiple_choices
        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][episode_idx]
            action['labels'] = [anno['multiple_choice_answer']]
        return action