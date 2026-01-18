from numba import types
from numba.core import config
class TestStructModelType(types.Type):

    def __init__(self):
        super().__init__(name='TestStructModelType')