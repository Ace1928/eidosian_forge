import warnings
from typing import cast, Sequence, Union, List, Tuple, Dict, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
def _get_quimb_version():
    """Returns the quimb version and parsed (major,minor) numbers if possible.
    Returns:
        a tuple of ((major, minor), version string)
    """
    version = quimb.__version__
    try:
        return (tuple((int(x) for x in version.split('.'))), version)
    except:
        return ((0, 0), version)