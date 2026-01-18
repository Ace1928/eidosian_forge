from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
from .build import build
import copy
import os
class SentencechooseTeacher(FbDialogTeacher):
    """
    Teacher for the sentence choosing task.

    Turkers were instructed to choose the 'most interesting' sentence from a paragraph.
    """

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _choose_sentence_path(opt)
        super().__init__(opt, shared)

    def next_example(self):
        action, epoch_done = super().next_example()
        action['label_candidates'] = list(action['label_candidates'])
        if '' in action['label_candidates']:
            action['label_candidates'].remove('')
        return (action, epoch_done)