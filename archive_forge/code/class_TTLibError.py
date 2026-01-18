from fontTools.misc.loggingTools import deprecateFunction
import logging
from fontTools.ttLib.ttFont import *
from fontTools.ttLib.ttCollection import TTCollection
class TTLibError(Exception):
    pass