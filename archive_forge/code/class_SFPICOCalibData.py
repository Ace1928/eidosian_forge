import os
from ...utils.filemanip import split_filename
from ..base import (
class SFPICOCalibData(StdOutCommandLine):
    """
    Generates Spherical Function PICo Calibration Data.

    SFPICOCalibData creates synthetic data for use with SFLUTGen. The
    synthetic data is generated using a mixture of gaussians, in the
    same way datasynth generates data.  Each voxel of data models a
    slightly different fibre configuration (varying FA and fibre-
    crossings) and undergoes a random rotation to help account for any
    directional bias in the chosen acquisition scheme.  A second file,
    which stores information about the datafile, is generated along with
    the datafile.

    Examples
    --------
    To create a calibration dataset using the default settings

    >>> import nipype.interfaces.camino as cam
    >>> calib = cam.SFPICOCalibData()
    >>> calib.inputs.scheme_file = 'A.scheme'
    >>> calib.inputs.snr = 20
    >>> calib.inputs.info_file = 'PICO_calib.info'
    >>> calib.run()           # doctest: +SKIP

    The default settings create a large dataset (249,231 voxels), of
    which 3401 voxels contain a single fibre population per voxel and
    the rest of the voxels contain two fibre-populations. The amount of
    data produced can be varied by specifying the ranges and steps of
    the parameters for both the one and two fibre datasets used.

    To create a custom calibration dataset

    >>> import nipype.interfaces.camino as cam
    >>> calib = cam.SFPICOCalibData()
    >>> calib.inputs.scheme_file = 'A.scheme'
    >>> calib.inputs.snr = 20
    >>> calib.inputs.info_file = 'PICO_calib.info'
    >>> calib.inputs.twodtfarange = [0.3, 0.9]
    >>> calib.inputs.twodtfastep = 0.02
    >>> calib.inputs.twodtanglerange = [0, 0.785]
    >>> calib.inputs.twodtanglestep = 0.03925
    >>> calib.inputs.twodtmixmax = 0.8
    >>> calib.inputs.twodtmixstep = 0.1
    >>> calib.run()              # doctest: +SKIP

    This would provide 76,313 voxels of synthetic data, where 3401 voxels
    simulate the one fibre cases and 72,912 voxels simulate the various
    two fibre cases. However, care should be taken to ensure that enough
    data is generated for calculating the LUT.      # doctest: +SKIP

    """
    _cmd = 'sfpicocalibdata'
    input_spec = SFPICOCalibDataInputSpec
    output_spec = SFPICOCalibDataOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['PICOCalib'] = os.path.abspath(self._gen_outfilename())
        outputs['calib_info'] = os.path.abspath(self.inputs.info_file)
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.scheme_file)
        return name + '_PICOCalib.Bfloat'