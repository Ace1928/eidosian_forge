from .base import FSLCommand, FSLCommandInputSpec
from ..base import TraitedSpec, File, traits
class B0Calc(FSLCommand):
    """
    B0 inhomogeneities occur at interfaces of materials with different magnetic susceptibilities,
    such as tissue-air interfaces. These differences lead to distortion in the local magnetic field,
    as Maxwell’s equations need to be satisfied. An example of B0 inhomogneity is the first volume
    of the 4D volume ```$FSLDIR/data/possum/b0_ppm.nii.gz```.

    Examples
    --------

    >>> from nipype.interfaces.fsl import B0Calc
    >>> b0calc = B0Calc()
    >>> b0calc.inputs.in_file = 'tissue+air_map.nii'
    >>> b0calc.inputs.z_b0 = 3.0
    >>> b0calc.inputs.output_type = "NIFTI_GZ"
    >>> b0calc.cmdline
    'b0calc -i tissue+air_map.nii -o tissue+air_map_b0field.nii.gz --chi0=4.000000e-07 -d -9.450000e-06 --extendboundary=1.00 --b0x=0.00 --gx=0.0000 --b0y=0.00 --gy=0.0000 --b0=3.00 --gz=0.0000'

    """
    _cmd = 'b0calc'
    input_spec = B0CalcInputSpec
    output_spec = B0CalcOutputSpec