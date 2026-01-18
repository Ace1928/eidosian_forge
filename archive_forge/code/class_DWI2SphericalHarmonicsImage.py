import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class DWI2SphericalHarmonicsImage(CommandLine):
    """
    Convert base diffusion-weighted images to their spherical harmonic representation.

    This program outputs the spherical harmonic decomposition for the set measured signal attenuations.
    The signal attenuations are calculated by identifying the b-zero images from the diffusion encoding supplied
    (i.e. those with zero as the b-value), and dividing the remaining signals by the mean b-zero signal intensity.
    The spherical harmonic decomposition is then calculated by least-squares linear fitting.
    Note that this program makes use of implied symmetries in the diffusion profile.

    First, the fact the signal attenuation profile is real implies that it has conjugate symmetry,
    i.e. Y(l,-m) = Y(l,m)* (where * denotes the complex conjugate). Second, the diffusion profile should be
    antipodally symmetric (i.e. S(x) = S(-x)), implying that all odd l components should be zero. Therefore,
    this program only computes the even elements.

    Note that the spherical harmonics equations used here differ slightly from those conventionally used,
    in that the (-1)^m factor has been omitted. This should be taken into account in all subsequent calculations.

    Each volume in the output image corresponds to a different spherical harmonic component, according to the following convention:

    * [0] Y(0,0)
    * [1] Im {Y(2,2)}
    * [2] Im {Y(2,1)}
    * [3] Y(2,0)
    * [4] Re {Y(2,1)}
    * [5] Re {Y(2,2)}
    * [6] Im {Y(4,4)}
    * [7] Im {Y(4,3)}

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> dwi2SH = mrt.DWI2SphericalHarmonicsImage()
    >>> dwi2SH.inputs.in_file = 'diffusion.nii'
    >>> dwi2SH.inputs.encoding_file = 'encoding.txt'
    >>> dwi2SH.run()                                    # doctest: +SKIP
    """
    _cmd = 'dwi2SH'
    input_spec = DWI2SphericalHarmonicsImageInputSpec
    output_spec = DWI2SphericalHarmonicsImageOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['spherical_harmonics_image'] = self.inputs.out_filename
        if not isdefined(outputs['spherical_harmonics_image']):
            outputs['spherical_harmonics_image'] = op.abspath(self._gen_outfilename())
        else:
            outputs['spherical_harmonics_image'] = op.abspath(outputs['spherical_harmonics_image'])
        return outputs

    def _gen_filename(self, name):
        if name == 'out_filename':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_SH.mif'