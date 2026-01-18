import warnings
import numpy as np
class FortranEOFError(TypeError, OSError):
    """Indicates that the file ended properly.

    This error descends from TypeError because the code used to raise
    TypeError (and this was the only way to know that the file had
    ended) so users might have ``except TypeError:``.

    """
    pass