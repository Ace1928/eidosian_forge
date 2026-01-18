from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class CloneMixin:

    def clone(self, **kwds):
        n = self.__class__()
        n.__dict__.clear()
        n.__dict__.update(self.__dict__)
        if kwds:
            n.__dict__.update(kwds)
        return n