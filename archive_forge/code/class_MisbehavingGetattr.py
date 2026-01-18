from IPython.utils.dir2 import dir2
import pytest
class MisbehavingGetattr:

    def __getattr__(self, attr):
        raise KeyError('I should be caught')

    def some_method(self):
        return True