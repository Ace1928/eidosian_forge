from parlai.core.teachers import ParlAIDialogTeacher
from parlai.utils.misc import warn_once
from .build import build
import copy
import os
class DistractorTeacher(ParlAIDialogTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt, 'distractor')
        super().__init__(opt, shared)