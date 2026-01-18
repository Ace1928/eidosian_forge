from urllib.parse import urlencode
from urllib.request import urlopen, Request
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.Align import Alignment
from Bio.Seq import reverse_complement
def __set_mask(self, mask):
    if self.length is None:
        self.__mask = ()
    elif mask is None:
        self.__mask = (1,) * self.length
    elif len(mask) != self.length:
        raise ValueError('The length (%d) of the mask is inconsistent with the length (%d) of the motif' % (len(mask), self.length))
    elif isinstance(mask, str):
        self.__mask = []
        for char in mask:
            if char == '*':
                self.__mask.append(1)
            elif char == ' ':
                self.__mask.append(0)
            else:
                raise ValueError("Mask should contain only '*' or ' ' and not a '%s'" % char)
        self.__mask = tuple(self.__mask)
    else:
        self.__mask = tuple((int(bool(c)) for c in mask))