import re
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping, MutableSequence
from warnings import warn
import numpy as np
from .. import xmlutils as xml
from ..arrayproxy import reshape_dataobj
from ..caret import CaretMetaData
from ..dataobj_images import DataobjImage
from ..filebasedimages import FileBasedHeader, SerializableImage
from ..nifti1 import Nifti1Extensions
from ..nifti2 import Nifti2Header, Nifti2Image
from ..volumeutils import Recoder, make_dt_codes
def _get_indices_from_mim(self, mim):
    applies_to_matrix_dimension = mim.applies_to_matrix_dimension
    if not isinstance(applies_to_matrix_dimension, Iterable):
        applies_to_matrix_dimension = (int(applies_to_matrix_dimension),)
    return applies_to_matrix_dimension