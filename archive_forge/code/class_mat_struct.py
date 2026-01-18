import numpy as np
from ._miobase import convert_dtypes
class mat_struct:
    """Placeholder for holding read data from structs.

    We use instances of this class when the user passes False as a value to the
    ``struct_as_record`` parameter of the :func:`scipy.io.loadmat` function.
    """
    pass