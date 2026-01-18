from parlai.core.teachers import FixedDialogTeacher
from .build import build
import json
import os
class MedHopTeacher(DefaultTeacher):

    def _path(self, opt):
        dt = opt['datatype'].split(':')
        datatype = 'train' if dt[0] == 'train' else 'dev'
        return os.path.join(opt['datapath'], 'qangaroo', 'qangaroo_v1.1', 'medhop', datatype + '.json')