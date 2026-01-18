import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class NIfTIDT2Camino(CommandLine):
    """
    Converts NIFTI-1 diffusion tensors to Camino format. The program reads the
    NIFTI header but does not apply any spatial transformations to the data. The
    NIFTI intensity scaling parameters are applied.

    The output is the tensors in Camino voxel ordering: [exit, ln(S0), dxx, dxy,
    dxz, dyy, dyz, dzz].

    The exit code is set to 0 unless a background mask is supplied, in which case
    the code is 0 in brain voxels and -1 in background voxels.

    The value of ln(S0) in the output is taken from a file if one is supplied,
    otherwise it is set to 0.

    NOTE FOR FSL USERS - FSL's dtifit can output NIFTI tensors, but they are not
    stored in the usual way (which is using NIFTI_INTENT_SYMMATRIX). FSL's
    tensors follow the ITK / VTK "upper-triangular" convention, so you will need
    to use the -uppertriangular option to convert these correctly.

    """
    _cmd = 'niftidt2camino'
    input_spec = NIfTIDT2CaminoInputSpec
    output_spec = NIfTIDT2CaminoOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_filename('out_file')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            _, filename, _ = split_filename(self.inputs.in_file)
        return filename