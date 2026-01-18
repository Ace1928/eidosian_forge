from parlai.core.teachers import FbDialogTeacher
from .build_2009 import build as build_2009
from .build_2018 import build as build_2018
import copy
import os
class V2018Task10kTeacher(Task10kTeacher):

    def __init__(self, opt, shared=None):
        super(V2018Task10kTeacher, self).__init__(opt, shared, '2018', True)