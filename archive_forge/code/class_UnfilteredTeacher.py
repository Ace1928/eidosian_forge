from parlai.core.teachers import FbDialogTeacher
from .build import build
import copy
import os
class UnfilteredTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        super().__init__(opt, shared)