import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
class WarpImageMultiTransform(ANTSCommand):
    """Warps an image from one space to another

    Examples
    --------

    >>> from nipype.interfaces.ants import WarpImageMultiTransform
    >>> wimt = WarpImageMultiTransform()
    >>> wimt.inputs.input_image = 'structural.nii'
    >>> wimt.inputs.reference_image = 'ants_deformed.nii.gz'
    >>> wimt.inputs.transformation_series = ['ants_Warp.nii.gz','ants_Affine.txt']
    >>> wimt.cmdline
    'WarpImageMultiTransform 3 structural.nii structural_wimt.nii -R ants_deformed.nii.gz ants_Warp.nii.gz ants_Affine.txt'

    >>> wimt = WarpImageMultiTransform()
    >>> wimt.inputs.input_image = 'diffusion_weighted.nii'
    >>> wimt.inputs.reference_image = 'functional.nii'
    >>> wimt.inputs.transformation_series = ['func2anat_coreg_Affine.txt','func2anat_InverseWarp.nii.gz',     'dwi2anat_Warp.nii.gz','dwi2anat_coreg_Affine.txt']
    >>> wimt.inputs.invert_affine = [1]  # this will invert the 1st Affine file: 'func2anat_coreg_Affine.txt'
    >>> wimt.cmdline
    'WarpImageMultiTransform 3 diffusion_weighted.nii diffusion_weighted_wimt.nii -R functional.nii -i func2anat_coreg_Affine.txt func2anat_InverseWarp.nii.gz dwi2anat_Warp.nii.gz dwi2anat_coreg_Affine.txt'

    """
    _cmd = 'WarpImageMultiTransform'
    input_spec = WarpImageMultiTransformInputSpec
    output_spec = WarpImageMultiTransformOutputSpec

    def _gen_filename(self, name):
        if name == 'output_image':
            _, name, ext = split_filename(os.path.abspath(self.inputs.input_image))
            return ''.join((name, self.inputs.out_postfix, ext))
        return None

    def _format_arg(self, opt, spec, val):
        if opt == 'transformation_series':
            series = []
            affine_counter = 0
            affine_invert = []
            for transformation in val:
                if 'affine' in transformation.lower() and isdefined(self.inputs.invert_affine):
                    affine_counter += 1
                    if affine_counter in self.inputs.invert_affine:
                        series += ['-i']
                        affine_invert.append(affine_counter)
                series += [transformation]
            if isdefined(self.inputs.invert_affine):
                diff_inv = set(self.inputs.invert_affine) - set(affine_invert)
                if diff_inv:
                    raise Exceptions('Review invert_affine, not all indexes from invert_affine were used, check the description for the full definition')
            return ' '.join(series)
        return super(WarpImageMultiTransform, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        if isdefined(self.inputs.output_image):
            outputs['output_image'] = os.path.abspath(self.inputs.output_image)
        else:
            outputs['output_image'] = os.path.abspath(self._gen_filename('output_image'))
        return outputs