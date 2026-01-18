import base64
import numpy as np
from . import _marching_cubes_lewiner_luts as mcluts
from . import _marching_cubes_lewiner_cy
def _to_array(args):
    shape, text = args
    byts = base64.decodebytes(text.encode('utf-8'))
    ar = np.frombuffer(byts, dtype='int8')
    ar.shape = shape
    return ar