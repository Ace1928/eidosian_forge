from copy import deepcopy as copy
from collections import namedtuple
import numpy as np
from .compat import filename_encode
from .datatype import Datatype
from .selections import SimpleSelection, select
from .. import h5d, h5p, h5s, h5t
class VDSmap(namedtuple('VDSmap', ('vspace', 'file_name', 'dset_name', 'src_space'))):
    """Defines a region in a virtual dataset mapping to part of a source dataset
    """