from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import copy
import os
class VTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        opt['datafile'] = _path('cbtest_V', opt)
        opt['cloze'] = True
        super().__init__(opt, shared)