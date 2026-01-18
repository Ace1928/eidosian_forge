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
class Cifti2MatrixIndicesMap(xml.XmlSerializable, MutableSequence):
    """Class for Matrix Indices Map

    * Description - Provides a mapping between matrix indices and their
      interpretation.
    * Attributes

        * AppliesToMatrixDimension - Lists the dimension(s) of the matrix to
          which this MatrixIndicesMap applies. The dimensions of the matrix
          start at zero (dimension 0 describes the indices along the first
          dimension, dimension 1 describes the indices along the second
          dimension, etc.). If this MatrixIndicesMap applies to more than one
          matrix dimension, the values are separated by a comma.
        * IndicesMapToDataType - Type of data to which the MatrixIndicesMap
          applies.
        * NumberOfSeriesPoints - Indicates how many samples there are in a
          series mapping type. For example, this could be the number of
          timepoints in a timeseries.
        * SeriesExponent - Integer, SeriesStart and SeriesStep must be
          multiplied by 10 raised to the power of the value of this attribute
          to give the actual values assigned to indices (e.g., if SeriesStart
          is "5" and SeriesExponent is "-3", the value of the first series
          point is 0.005).
        * SeriesStart - Indicates what quantity should be assigned to the first
          series point.
        * SeriesStep - Indicates amount of change between each series point.
        * SeriesUnit - Indicates the unit of the result of multiplying
          SeriesStart and SeriesStep by 10 to the power of SeriesExponent.

    * Child Elements

        * BrainModel (0...N)
        * NamedMap (0...N)
        * Parcel (0...N)
        * Surface (0...N)
        * Volume (0...1)

    * Text Content: [NA]
    * Parent Element - Matrix

    Attributes
    ----------
    applies_to_matrix_dimension : list of ints
        Dimensions of this matrix that follow this mapping
    indices_map_to_data_type : str one of CIFTI_MAP_TYPES
        Type of mapping to the matrix indices
    number_of_series_points : int, optional
        If it is a series, number of points in the series
    series_exponent : int, optional
        If it is a series the exponent of the increment
    series_start : float, optional
        If it is a series, starting time
    series_step : float, optional
        If it is a series, step per element
    series_unit : str, optional
        If it is a series, units
    """
    _valid_type_mappings_ = {Cifti2BrainModel: ('CIFTI_INDEX_TYPE_BRAIN_MODELS',), Cifti2Parcel: ('CIFTI_INDEX_TYPE_PARCELS',), Cifti2NamedMap: ('CIFTI_INDEX_TYPE_LABELS',), Cifti2Volume: ('CIFTI_INDEX_TYPE_SCALARS', 'CIFTI_INDEX_TYPE_SERIES'), Cifti2Surface: ('CIFTI_INDEX_TYPE_SCALARS', 'CIFTI_INDEX_TYPE_SERIES')}

    def __init__(self, applies_to_matrix_dimension, indices_map_to_data_type, number_of_series_points=None, series_exponent=None, series_start=None, series_step=None, series_unit=None, maps=[]):
        self.applies_to_matrix_dimension = applies_to_matrix_dimension
        self.indices_map_to_data_type = indices_map_to_data_type
        self.number_of_series_points = number_of_series_points
        self.series_exponent = series_exponent
        self.series_start = series_start
        self.series_step = series_step
        self.series_unit = series_unit
        self._maps = []
        for m in maps:
            self.append(m)

    def __len__(self):
        return len(self._maps)

    def __delitem__(self, index):
        del self._maps[index]

    def __getitem__(self, index):
        return self._maps[index]

    def __setitem__(self, index, value):
        if isinstance(value, Cifti2Volume) and (self.volume is not None and (not isinstance(self._maps[index], Cifti2Volume))):
            raise Cifti2HeaderError('Only one Volume can be in a MatrixIndicesMap')
        self._maps[index] = value

    def insert(self, index, value):
        if isinstance(value, Cifti2Volume) and self.volume is not None:
            raise Cifti2HeaderError('Only one Volume can be in a MatrixIndicesMap')
        self._maps.insert(index, value)

    @property
    def named_maps(self):
        for p in self:
            if isinstance(p, Cifti2NamedMap):
                yield p

    @property
    def surfaces(self):
        for p in self:
            if isinstance(p, Cifti2Surface):
                yield p

    @property
    def parcels(self):
        for p in self:
            if isinstance(p, Cifti2Parcel):
                yield p

    @property
    def volume(self):
        for p in self:
            if isinstance(p, Cifti2Volume):
                return p
        return None

    @volume.setter
    def volume(self, volume):
        if not isinstance(volume, Cifti2Volume):
            raise ValueError('You can only set a volume with a volume')
        for i, v in enumerate(self):
            if isinstance(v, Cifti2Volume):
                break
        else:
            self.append(volume)
            return
        self[i] = volume

    @volume.deleter
    def volume(self):
        for i, v in enumerate(self):
            if isinstance(v, Cifti2Volume):
                break
        else:
            raise ValueError('No Cifti2Volume element')
        del self[i]

    @property
    def brain_models(self):
        for p in self:
            if isinstance(p, Cifti2BrainModel):
                yield p

    def _to_xml_element(self):
        if self.applies_to_matrix_dimension is None:
            raise Cifti2HeaderError('MatrixIndicesMap element requires to be applied to at least 1 dimension')
        mat_ind_map = xml.Element('MatrixIndicesMap')
        dims_as_strings = [str(dim) for dim in self.applies_to_matrix_dimension]
        mat_ind_map.attrib['AppliesToMatrixDimension'] = ','.join(dims_as_strings)
        for key in ('IndicesMapToDataType', 'NumberOfSeriesPoints', 'SeriesExponent', 'SeriesStart', 'SeriesStep', 'SeriesUnit'):
            attr = _underscore(key)
            value = getattr(self, attr)
            if value is not None:
                mat_ind_map.attrib[key] = str(value)
        for map_ in self:
            mat_ind_map.append(map_._to_xml_element())
        return mat_ind_map