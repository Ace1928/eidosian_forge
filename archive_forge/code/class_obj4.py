import dill
from functools import partial
import warnings
class obj4(object):

    def __init__(self):
        super(obj4, self).__init__()
        a = self

        class obj5(object):

            def __init__(self):
                super(obj5, self).__init__()
                self.a = a
        self.b = obj5()