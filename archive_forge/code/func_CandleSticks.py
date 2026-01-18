from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
def CandleSticks(**kwds):
    return TypedPropertyCollection(CandleStickProperties, **kwds)