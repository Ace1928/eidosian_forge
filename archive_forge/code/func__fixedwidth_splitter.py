import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def _fixedwidth_splitter(self, line):
    if self.comments is not None:
        line = line.split(self.comments)[0]
    line = line.strip('\r\n')
    if not line:
        return []
    fixed = self.delimiter
    slices = [slice(i, i + fixed) for i in range(0, len(line), fixed)]
    return [line[s] for s in slices]