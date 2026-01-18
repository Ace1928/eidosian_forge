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
class Cifti2Matrix(xml.XmlSerializable, MutableSequence):
    """CIFTI-2 Matrix object

    This is a list-like container where the elements are instances of
    :class:`Cifti2MatrixIndicesMap`.

    * Description: contains child elements that describe the meaning of the
      values in the matrix.
    * Attributes: [NA]
    * Child Elements

        * MetaData (0 .. 1)
        * MatrixIndicesMap (1 .. N)

    * Text Content: [NA]
    * Parent Element: CIFTI

    For each matrix (data) dimension, exactly one MatrixIndicesMap element must
    list it in the AppliesToMatrixDimension attribute.
    """

    def __init__(self):
        self._mims = []
        self.metadata = None

    @property
    def metadata(self):
        return self._meta

    @metadata.setter
    def metadata(self, meta):
        """Set the metadata for this Cifti2Header

        Parameters
        ----------
        meta : Cifti2MetaData

        Returns
        -------
        None
        """
        self._meta = _value_if_klass(meta, Cifti2MetaData)

    def _get_indices_from_mim(self, mim):
        applies_to_matrix_dimension = mim.applies_to_matrix_dimension
        if not isinstance(applies_to_matrix_dimension, Iterable):
            applies_to_matrix_dimension = (int(applies_to_matrix_dimension),)
        return applies_to_matrix_dimension

    @property
    def mapped_indices(self):
        """
        List of matrix indices that are mapped
        """
        mapped_indices = []
        for v in self:
            a2md = self._get_indices_from_mim(v)
            mapped_indices += a2md
        return mapped_indices

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
        for v in self:
            a2md = self._get_indices_from_mim(v)
            if index in a2md:
                return v
        raise Cifti2HeaderError('Index not mapped')

    def _validate_new_mim(self, value):
        if value.applies_to_matrix_dimension is None:
            raise Cifti2HeaderError('Cifti2MatrixIndicesMap needs to have the applies_to_matrix_dimension attribute set')
        a2md = self._get_indices_from_mim(value)
        if not set(self.mapped_indices).isdisjoint(a2md):
            raise Cifti2HeaderError('Indices in this Cifti2MatrixIndicesMap already mapped in this matrix')

    def __setitem__(self, key, value):
        if not isinstance(value, Cifti2MatrixIndicesMap):
            raise TypeError('Not a valid Cifti2MatrixIndicesMap instance')
        self._validate_new_mim(value)
        self._mims[key] = value

    def __getitem__(self, key):
        return self._mims[key]

    def __delitem__(self, key):
        del self._mims[key]

    def __len__(self):
        return len(self._mims)

    def insert(self, index, value):
        if not isinstance(value, Cifti2MatrixIndicesMap):
            raise TypeError('Not a valid Cifti2MatrixIndicesMap instance')
        self._validate_new_mim(value)
        self._mims.insert(index, value)

    def _to_xml_element(self):
        mat = xml.Element('Matrix')
        if self.metadata:
            mat.append(self.metadata._to_xml_element())
        for mim in self._mims:
            mat.append(mim._to_xml_element())
        return mat

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
        from . import cifti2_axes
        return cifti2_axes.from_index_mapping(self.get_index_map(index))

    def get_data_shape(self):
        """
        Returns data shape expected based on the CIFTI-2 header

        Any dimensions omitted in the CIFTI-2 header will be given a default size of None.
        """
        from . import cifti2_axes
        if len(self.mapped_indices) == 0:
            return ()
        base_shape = [None] * (max(self.mapped_indices) + 1)
        for mim in self:
            size = len(cifti2_axes.from_index_mapping(mim))
            for idx in mim.applies_to_matrix_dimension:
                base_shape[idx] = size
        return tuple(base_shape)