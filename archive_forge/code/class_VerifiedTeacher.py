from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from .build import build
import copy
import json
import os
class VerifiedTeacher(MultiTaskTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = 'triviaqa:VerifiedWikipedia,triviaqa:VerifiedWeb'
        super().__init__(opt, shared)