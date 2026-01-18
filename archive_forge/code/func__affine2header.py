from os.path import splitext
import numpy as np
from ..affines import from_matvec, voxel_sizes
from ..arrayproxy import ArrayProxy, reshape_dataobj
from ..batteryrunners import BatteryRunner, Report
from ..filebasedimages import SerializableImage
from ..fileholders import FileHolder
from ..filename_parser import _stringify_path
from ..openers import ImageOpener
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..volumeutils import Recoder, array_from_file, array_to_file, endian_codes
from ..wrapstruct import LabeledWrapStruct
def _affine2header(self):
    """Unconditionally set affine into the header"""
    hdr = self._header
    shape = np.array(self._dataobj.shape[:3])
    voxelsize = voxel_sizes(self._affine)
    Mdc = self._affine[:3, :3] / voxelsize
    c_ras = self._affine.dot(np.hstack((shape / 2.0, [1])))[:3]
    hdr['delta'] = voxelsize
    hdr['Mdc'] = Mdc.T
    hdr['Pxyz_c'] = c_ras