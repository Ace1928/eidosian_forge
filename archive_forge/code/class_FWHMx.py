import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class FWHMx(AFNICommandBase):
    """
    Unlike the older 3dFWHM, this program computes FWHMs for all sub-bricks
    in the input dataset, each one separately.  The output for each one is
    written to the file specified by '-out'.  The mean (arithmetic or geometric)
    of all the FWHMs along each axis is written to stdout.  (A non-positive
    output value indicates something bad happened; e.g., FWHM in z is meaningless
    for a 2D dataset; the estimation method computed incoherent intermediate results.)

    For complete details, see the `3dFWHMx Documentation.
    <https://afni.nimh.nih.gov/pub../pub/dist/doc/program_help/3dFWHMx.html>`_

    (Classic) METHOD:

      * Calculate ratio of variance of first differences to data variance.
      * Should be the same as 3dFWHM for a 1-brick dataset.
        (But the output format is simpler to use in a script.)


    .. note:: IMPORTANT NOTE [AFNI > 16]

      A completely new method for estimating and using noise smoothness values is
      now available in 3dFWHMx and 3dClustSim. This method is implemented in the
      '-acf' options to both programs.  'ACF' stands for (spatial) AutoCorrelation
      Function, and it is estimated by calculating moments of differences out to
      a larger radius than before.

      Notably, real FMRI data does not actually have a Gaussian-shaped ACF, so the
      estimated ACF is then fit (in 3dFWHMx) to a mixed model (Gaussian plus
      mono-exponential) of the form

        .. math::

          ACF(r) = a * exp(-r*r/(2*b*b)) + (1-a)*exp(-r/c)


      where :math:`r` is the radius, and :math:`a, b, c` are the fitted parameters.
      The apparent FWHM from this model is usually somewhat larger in real data
      than the FWHM estimated from just the nearest-neighbor differences used
      in the 'classic' analysis.

      The longer tails provided by the mono-exponential are also significant.
      3dClustSim has also been modified to use the ACF model given above to generate
      noise random fields.

    .. note:: TL;DR or summary

      The take-awaymessage is that the 'classic' 3dFWHMx and
      3dClustSim analysis, using a pure Gaussian ACF, is not very correct for
      FMRI data -- I cannot speak for PET or MEG data.

    .. warning::

      Do NOT use 3dFWHMx on the statistical results (e.g., '-bucket') from
      3dDeconvolve or 3dREMLfit!!!  The function of 3dFWHMx is to estimate
      the smoothness of the time series NOISE, not of the statistics. This
      proscription is especially true if you plan to use 3dClustSim next!!

    .. note:: Recommendations

      * For FMRI statistical purposes, you DO NOT want the FWHM to reflect
        the spatial structure of the underlying anatomy.  Rather, you want
        the FWHM to reflect the spatial structure of the noise.  This means
        that the input dataset should not have anatomical (spatial) structure.
      * One good form of input is the output of '3dDeconvolve -errts', which is
        the dataset of residuals left over after the GLM fitted signal model is
        subtracted out from each voxel's time series.
      * If you don't want to go to that much trouble, use '-detrend' to approximately
        subtract out the anatomical spatial structure, OR use the output of 3dDetrend
        for the same purpose.
      * If you do not use '-detrend', the program attempts to find non-zero spatial
        structure in the input, and will print a warning message if it is detected.

    .. note:: Notes on -demend

      * I recommend this option, and it is not the default only for historical
        compatibility reasons.  It may become the default someday.
      * It is already the default in program 3dBlurToFWHM. This is the same detrending
        as done in 3dDespike; using 2*q+3 basis functions for q > 0.
      * If you don't use '-detrend', the program now [Aug 2010] checks if a large number
        of voxels are have significant nonzero means. If so, the program will print a
        warning message suggesting the use of '-detrend', since inherent spatial
        structure in the image will bias the estimation of the FWHM of the image time
        series NOISE (which is usually the point of using 3dFWHMx).

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> fwhm = afni.FWHMx()
    >>> fwhm.inputs.in_file = 'functional.nii'
    >>> fwhm.cmdline
    '3dFWHMx -input functional.nii -out functional_subbricks.out > functional_fwhmx.out'
    >>> res = fwhm.run()  # doctest: +SKIP

    """
    _cmd = '3dFWHMx'
    input_spec = FWHMxInputSpec
    output_spec = FWHMxOutputSpec
    _references = [{'entry': BibTeX('@article{CoxReynoldsTaylor2016,author={R.W. Cox, R.C. Reynolds, and P.A. Taylor},title={AFNI and clustering: false positive rates redux},journal={bioRxiv},year={2016},}'), 'tags': ['method']}]
    _acf = True

    def _parse_inputs(self, skip=None):
        if not self.inputs.detrend:
            if skip is None:
                skip = []
            skip += ['out_detrend']
        return super(FWHMx, self)._parse_inputs(skip=skip)

    def _format_arg(self, name, trait_spec, value):
        if name == 'detrend':
            if value is True:
                return trait_spec.argstr
            elif value is False:
                return None
            elif isinstance(value, int):
                return trait_spec.argstr + ' %d' % value
        if name == 'acf':
            if value is True:
                return trait_spec.argstr
            elif value is False:
                self._acf = False
                return None
            elif isinstance(value, tuple):
                return trait_spec.argstr + ' %s %f' % value
            elif isinstance(value, (str, bytes)):
                return trait_spec.argstr + ' ' + value
        return super(FWHMx, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = super(FWHMx, self)._list_outputs()
        if self.inputs.detrend:
            fname, ext = op.splitext(self.inputs.in_file)
            if '.gz' in ext:
                _, ext2 = op.splitext(fname)
                ext = ext2 + ext
            outputs['out_detrend'] += ext
        else:
            outputs['out_detrend'] = Undefined
        sout = np.loadtxt(outputs['out_file'])
        if sout.size == 8:
            outputs['fwhm'] = tuple(sout[0, :])
        else:
            outputs['fwhm'] = tuple(sout)
        if self._acf:
            assert sout.size == 8, 'Wrong number of elements in %s' % str(sout)
            outputs['acf_param'] = tuple(sout[1])
            outputs['out_acf'] = op.abspath('3dFWHMx.1D')
            if isinstance(self.inputs.acf, (str, bytes)):
                outputs['out_acf'] = op.abspath(self.inputs.acf)
        return outputs