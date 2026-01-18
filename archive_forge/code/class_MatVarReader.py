import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
class MatVarReader:
    """ Abstract class defining required interface for var readers"""

    def __init__(self, file_reader):
        pass

    def read_header(self):
        """ Returns header """
        pass

    def array_from_header(self, header):
        """ Reads array given header """
        pass