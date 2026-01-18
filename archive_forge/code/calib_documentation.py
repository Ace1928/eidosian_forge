import os
from ...utils.filemanip import split_filename
from ..base import (

    Generates PICo lookup tables (LUT) for multi-fibre methods such as
    PASMRI and Q-Ball.

    SFLUTGen creates the lookup tables for the generalized multi-fibre
    implementation of the PICo tractography algorithm.  The outputs of
    this utility are either surface or line coefficients up to a given
    order. The calibration can be performed for different distributions,
    such as the Bingham and Watson distributions.

    This utility uses calibration data generated from SFPICOCalibData
    and peak information created by SFPeaks.

    The utility outputs two lut's, ``*_oneFibreSurfaceCoeffs.Bdouble`` and
    ``*_twoFibreSurfaceCoeffs.Bdouble``. Each of these files contains big-endian doubles
    as standard. The format of the output is::

          dimensions    (1 for Watson, 2 for Bingham)
          order         (the order of the polynomial)
          coefficient_1
          coefficient_2
          ...
          coefficient_N

    In  the case of the Watson, there is a single set of coefficients,
    which are ordered::

          constant, x, x^2, ..., x^order.

    In the case of the Bingham, there are two sets of coefficients (one
    for each surface), ordered so that::

          for j = 1 to order
            for k = 1 to order
              coeff_i = x^j * y^k
          where j+k < order

    Example
    -------
    To create a calibration dataset using the default settings

    >>> import nipype.interfaces.camino as cam
    >>> lutgen = cam.SFLUTGen()
    >>> lutgen.inputs.in_file = 'QSH_peaks.Bdouble'
    >>> lutgen.inputs.info_file = 'PICO_calib.info'
    >>> lutgen.run()        # doctest: +SKIP

    