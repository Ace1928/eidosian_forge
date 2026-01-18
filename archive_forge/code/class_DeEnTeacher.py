from parlai.core.teachers import DialogTeacher
from .build import build
import copy
import os
class DeEnTeacher(DefaultTeacher):

    def __init__(self, opt, shared=None):
        opt['task'] = 'iwslt14:de_en'
        super().__init__(opt, shared)