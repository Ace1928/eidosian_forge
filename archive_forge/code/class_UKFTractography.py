import os
from ....base import (
class UKFTractography(SEMLikeCommandLine):
    """title: UKF Tractography

    category: Diffusion.Tractography

    description: This module traces fibers in a DWI Volume using the multiple tensor unscented Kalman Filter methology. For more information check the documentation.

    version: 1.0

    documentation-url: http://www.nitrc.org/plugins/mwiki/index.php/ukftractography:MainPage

    contributor: Yogesh Rathi, Stefan Lienhard, Yinpeng Li, Martin Styner, Ipek Oguz, Yundi Shi, Christian Baumgartner, Kent Williams, Hans Johnson, Peter Savadjiev, Carl-Fredrik Westin.

    acknowledgements: The development of this module was supported by NIH grants R01 MH097979 (PI Rathi), R01 MH092862 (PIs Westin and Verma), U01 NS083223 (PI Westin), R01 MH074794 (PI Westin) and P41 EB015902 (PI Kikinis).
    """
    input_spec = UKFTractographyInputSpec
    output_spec = UKFTractographyOutputSpec
    _cmd = ' UKFTractography '
    _outputs_filenames = {'tracts': 'tracts.vtp', 'tractsWithSecondTensor': 'tractsWithSecondTensor.vtp'}
    _redirect_x = False