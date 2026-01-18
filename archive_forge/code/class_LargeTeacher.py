from parlai.core.teachers import Teacher
from .build import build
import json
import os
import random
class LargeTeacher(TaskNTalkTeacher):
    """
    Teacher for large dataset, invoked by ``taskntalk:large``.
    """

    def __init__(self, opt, shared=None):
        opt['datafile'] = _path(opt, 'large')
        super().__init__(opt, shared)