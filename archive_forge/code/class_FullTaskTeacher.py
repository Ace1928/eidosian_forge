from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import copy
import os
class FullTaskTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        opt['datafile'] = _path('full', opt['task'].split(':')[2], opt)
        opt['cands_datafile'] = os.path.join(opt['datapath'], 'personalized-dialog', 'personalized-dialog-dataset', 'personalized-dialog-candidates.txt')
        super().__init__(opt, shared)