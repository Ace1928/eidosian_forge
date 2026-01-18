from IPython.core import autocall
from IPython.testing import tools as tt
class Autocallable(autocall.IPyAutocall):

    def __call__(self):
        return 'called'