from . import number_types as N
from . import packer
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
def GetVectorAsNumpy(numpy_type, buf, count, offset):
    """ GetVecAsNumpy decodes values starting at buf[head] as
    `numpy_type`, where `numpy_type` is a numpy dtype. """
    if np is not None:
        return np.frombuffer(buf, dtype=numpy_type, count=count, offset=offset)
    else:
        raise NumpyRequiredForThisFeature('Numpy was not found.')