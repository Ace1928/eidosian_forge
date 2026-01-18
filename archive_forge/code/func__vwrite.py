import skvideo.io
import sys
import numpy as np
import hashlib
import os
from numpy.testing import assert_equal
def _vwrite(backend):
    outputfile = sys._getframe().f_code.co_name + '.mp4'
    np.random.seed(0)
    outputdata = np.random.random(size=(5, 480, 640, 3)) * 255
    outputdata = outputdata.astype(np.uint8)
    skvideo.io.vwrite(outputfile, outputdata, backend=backend)
    h = hashfile(open(outputfile, 'rb'), hashlib.sha256())
    os.remove(outputfile)