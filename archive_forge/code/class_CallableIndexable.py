from IPython.core import autocall
from IPython.testing import tools as tt
class CallableIndexable(object):

    def __getitem__(self, idx):
        return True

    def __call__(self, *args, **kws):
        return True