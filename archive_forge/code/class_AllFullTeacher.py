from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import copy
import os
class AllFullTeacher(MultiTaskTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(('personalized_dialog:FullTask:%d' % (i + 1) for i in range(5)))
        opt['cands_datafile'] = os.path.join(opt['datapath'], 'personalized-dialog', 'personalized-dialog-dataset', 'personalized-dialog-candidates.txt')
        super().__init__(opt, shared)