import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
class EpsDSC(PsDSC):

    def __init__(self):
        PsDSC.__init__(self)

    def documentHeader(self):
        return '%!PS-Adobe-3.0 EPSF-3.0'