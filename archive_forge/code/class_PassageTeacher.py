import copy
import json
import os
from parlai.core.teachers import DialogTeacher, FbDialogTeacher
from .build import build
class PassageTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, is_passage=True)
        super().__init__(opt, shared)