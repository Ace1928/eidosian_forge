from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build
import copy
import json
import os
class FulldocTeacher(ParlAIDialogTeacher):

    def __init__(self, opt, shared=None):
        build(opt)
        opt = copy.deepcopy(opt)
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'valid'
        datafile = os.path.join(opt['datapath'], 'SQuAD-fulldoc', 'squad_fulldocs.' + suffix + ':ordered')
        opt['parlaidialogteacher_datafile'] = datafile
        super().__init__(opt, shared)
        self.id = 'squad-fulldoc'
        self.reset()