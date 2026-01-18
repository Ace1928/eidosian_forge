from collections import OrderedDict
import warnings
from ._util import get_backend
from .chemistry import Substance
from .units import allclose
class LimitingDebyeHuckelActivityProduct(_ActivityProductBase):

    def __call__(self, c):
        z = self.args[0]
        IS = ionic_strength(c, z)
        return limiting_activity_product(IS, self.stoich, *self.args)