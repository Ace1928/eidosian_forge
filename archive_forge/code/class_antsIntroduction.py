from builtins import range
import os
from glob import glob
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, OutputMultiPath
from ...utils.filemanip import split_filename
class antsIntroduction(ANTSCommand):
    """Uses ANTS to generate matrices to warp data from one space to another.

    Examples
    --------

    >>> from nipype.interfaces.ants.legacy import antsIntroduction
    >>> warp = antsIntroduction()
    >>> warp.inputs.reference_image = 'Template_6.nii'
    >>> warp.inputs.input_image = 'structural.nii'
    >>> warp.inputs.max_iterations = [30,90,20]
    >>> warp.cmdline
    'antsIntroduction.sh -d 3 -i structural.nii -m 30x90x20 -o ants_ -r Template_6.nii -t GR'

    """
    _cmd = 'antsIntroduction.sh'
    input_spec = antsIntroductionInputSpec
    output_spec = antsIntroductionOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        transmodel = self.inputs.transformation_model
        if not isdefined(transmodel) or (isdefined(transmodel) and transmodel not in ['RI', 'RA']):
            outputs['warp_field'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'Warp.nii.gz')
            outputs['inverse_warp_field'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'InverseWarp.nii.gz')
        outputs['affine_transformation'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'Affine.txt')
        outputs['input_file'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'repaired.nii.gz')
        outputs['output_file'] = os.path.join(os.getcwd(), self.inputs.out_prefix + 'deformed.nii.gz')
        return outputs