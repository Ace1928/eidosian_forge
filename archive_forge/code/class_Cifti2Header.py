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
class Cifti2Header(FileBasedHeader, xml.XmlSerializable):
    """Class for CIFTI-2 header extension"""

    def __init__(self, matrix=None, version='2.0'):
        FileBasedHeader.__init__(self)
        xml.XmlSerializable.__init__(self)
        if matrix is None:
            matrix = Cifti2Matrix()
        self.matrix = matrix
        self.version = version

    def _to_xml_element(self):
        cifti = xml.Element('CIFTI')
        cifti.attrib['Version'] = str(self.version)
        mat_xml = self.matrix._to_xml_element()
        if mat_xml is not None:
            cifti.append(mat_xml)
        return cifti

    def __eq__(self, other):
        return self.to_xml() == other.to_xml()

    @classmethod
    def may_contain_header(klass, binaryblock):
        from .parse_cifti2 import _Cifti2AsNiftiHeader
        return _Cifti2AsNiftiHeader.may_contain_header(binaryblock)

    @property
    def number_of_mapped_indices(self):
        """
        Number of mapped indices
        """
        return len(self.matrix)

    @property
    def mapped_indices(self):
        """
        List of matrix indices that are mapped
        """
        return self.matrix.mapped_indices

    def get_index_map(self, index):
        """
        Cifti2 Mapping class for a given index

        Parameters
        ----------
        index : int
            Index for which we want to obtain the mapping.
            Must be in the mapped_indices sequence.

        Returns
        -------
        cifti2_map : Cifti2MatrixIndicesMap
            Returns the Cifti2MatrixIndicesMap corresponding to
            the given index.
        """
        return self.matrix.get_index_map(index)

    def get_axis(self, index):
        """
        Generates the Cifti2 axis for a given dimension

        Parameters
        ----------
        index : int
            Dimension for which we want to obtain the mapping.

        Returns
        -------
        axis : :class:`.cifti2_axes.Axis`
        """
        return self.matrix.get_axis(index)

    @classmethod
    def from_axes(cls, axes):
        """
        Creates a new Cifti2 header based on the Cifti2 axes

        Parameters
        ----------
        axes : tuple of :class`.cifti2_axes.Axis`
            sequence of Cifti2 axes describing each row/column of the matrix to be stored

        Returns
        -------
        header : Cifti2Header
            new header describing the rows/columns in a format consistent with Cifti2
        """
        from . import cifti2_axes
        return cifti2_axes.to_header(axes)