from parlai.core.teachers import Teacher
from .build import build
import json
import os
import random
class SmallTeacher(TaskNTalkTeacher):
    """
    Teacher for small dataset, invoked by ``taskntalk:small``.
    """

    def __init__(self, opt, shared=None):
        opt['datafile'] = _path(opt, 'small')
        super().__init__(opt, shared)