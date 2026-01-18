import dill
from functools import partial
import warnings
class obj3(object):
    super_ = super

    def __init__(self):
        obj3.super_(obj3, self).__init__()