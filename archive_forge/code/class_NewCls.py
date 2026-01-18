from functools import partialmethod
from torch import optim
class NewCls(cls):
    __init__ = partialmethod(cls.__init__, *args, **kwargs)