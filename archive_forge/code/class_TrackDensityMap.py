import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, BaseInterfaceInputSpec, File, isdefined, traits
from .base import (
class TrackDensityMap(DipyBaseInterface):
    """
    Creates a tract density image from a TrackVis track file using functions
    from dipy

    Example
    -------

    >>> import nipype.interfaces.dipy as dipy
    >>> trk2tdi = dipy.TrackDensityMap()
    >>> trk2tdi.inputs.in_file = 'converted.trk'
    >>> trk2tdi.run()                                   # doctest: +SKIP

    """
    input_spec = TrackDensityMapInputSpec
    output_spec = TrackDensityMapOutputSpec

    def _run_interface(self, runtime):
        from numpy import min_scalar_type
        from dipy.tracking.utils import density_map
        import nibabel.trackvis as nbt
        tracks, header = nbt.read(self.inputs.in_file)
        streams = (ii[0] for ii in tracks)
        if isdefined(self.inputs.reference):
            refnii = nb.load(self.inputs.reference)
            affine = refnii.affine
            data_dims = refnii.shape[:3]
            kwargs = dict(affine=affine)
        else:
            IFLOGGER.warning('voxel_dims and data_dims are deprecated as of dipy 0.7.1. Please use reference input instead')
            if not isdefined(self.inputs.data_dims):
                data_dims = header['dim']
            else:
                data_dims = self.inputs.data_dims
            if not isdefined(self.inputs.voxel_dims):
                voxel_size = header['voxel_size']
            else:
                voxel_size = self.inputs.voxel_dims
            affine = header['vox_to_ras']
            kwargs = dict(voxel_size=voxel_size)
        data = density_map(streams, data_dims, **kwargs)
        data = data.astype(min_scalar_type(data.max()))
        img = nb.Nifti1Image(data, affine)
        out_file = op.abspath(self.inputs.out_filename)
        nb.save(img, out_file)
        IFLOGGER.info('Track density map saved as %s, size=%s, dimensions=%s', out_file, img.shape, img.header.get_zooms())
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = op.abspath(self.inputs.out_filename)
        return outputs