from functools import partial
from pytest import raises
from ..argument import Argument
from ..field import Field
from ..scalars import String
from ..structures import NonNull
from .utils import MyLazyType
class MyInstance:
    value = 'value'
    value_func = staticmethod(lambda: 'value_func')

    def value_method(self):
        return 'value_method'