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
class Cifti2VertexIndices(xml.XmlSerializable, MutableSequence):
    """CIFTI-2 vertex indices: vertex indices for an associated brain model

    The vertex indices (which are independent for each surface, and
    zero-based) that are used in this brain model[.] The parent
    BrainModel's ``index_count`` indicates the number of indices.

    * Description - Contains a list of vertex indices for a BrainModel with
      ModelType equal to CIFTI_MODEL_TYPE_SURFACE.
    * Attributes: [NA]
    * Child Elements: [NA]
    * Text Content - The vertex indices (which are independent for each
      surface, and zero-based) that are used in this brain model, with each
      index separated by a whitespace character.  The parent BrainModel's
      IndexCount attribute indicates the number of indices in this element's
      content.
    * Parent Element - BrainModel
    """

    def __init__(self, indices=None):
        self._indices = []
        if indices is not None:
            self.extend(indices)

    def __len__(self):
        return len(self._indices)

    def __delitem__(self, index):
        del self._indices[index]

    def __getitem__(self, index):
        return self._indices[index]

    def __setitem__(self, index, value):
        try:
            value = int(value)
            self._indices[index] = value
        except ValueError:
            raise ValueError('value must be an int')

    def insert(self, index, value):
        try:
            value = int(value)
            self._indices.insert(index, value)
        except ValueError:
            raise ValueError('value must be an int')

    def _to_xml_element(self):
        if len(self) == 0:
            raise Cifti2HeaderError('VertexIndices element requires indices')
        vert_indices = xml.Element('VertexIndices')
        vert_indices.text = ' '.join([str(i) for i in self])
        return vert_indices