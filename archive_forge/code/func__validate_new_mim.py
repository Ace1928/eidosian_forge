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
def _validate_new_mim(self, value):
    if value.applies_to_matrix_dimension is None:
        raise Cifti2HeaderError('Cifti2MatrixIndicesMap needs to have the applies_to_matrix_dimension attribute set')
    a2md = self._get_indices_from_mim(value)
    if not set(self.mapped_indices).isdisjoint(a2md):
        raise Cifti2HeaderError('Indices in this Cifti2MatrixIndicesMap already mapped in this matrix')