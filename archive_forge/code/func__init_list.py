from sympy.multipledispatch.dispatcher import (Dispatcher, MDNotImplementedError,
from sympy.testing.pytest import raises, warns
@__init__.register(list)
def _init_list(self, data):
    self.data = data