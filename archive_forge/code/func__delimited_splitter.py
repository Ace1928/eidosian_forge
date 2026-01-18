import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def _delimited_splitter(self, line):
    """Chop off comments, strip, and split at delimiter. """
    if self.comments is not None:
        line = line.split(self.comments)[0]
    line = line.strip(' \r\n')
    if not line:
        return []
    return line.split(self.delimiter)