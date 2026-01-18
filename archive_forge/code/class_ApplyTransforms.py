import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
class ApplyTransforms(ANTSCommand):
    """ApplyTransforms, applied to an input image, transforms it according to a
    reference image and a transform (or a set of transforms).

    Examples
    --------

    >>> from nipype.interfaces.ants import ApplyTransforms
    >>> at = ApplyTransforms()
    >>> at.inputs.input_image = 'moving1.nii'
    >>> at.inputs.reference_image = 'fixed1.nii'
    >>> at.inputs.transforms = 'identity'
    >>> at.cmdline
    'antsApplyTransforms --default-value 0 --float 0 --input moving1.nii --interpolation Linear --output moving1_trans.nii --reference-image fixed1.nii --transform identity'

    >>> at = ApplyTransforms()
    >>> at.inputs.dimension = 3
    >>> at.inputs.input_image = 'moving1.nii'
    >>> at.inputs.reference_image = 'fixed1.nii'
    >>> at.inputs.output_image = 'deformed_moving1.nii'
    >>> at.inputs.interpolation = 'Linear'
    >>> at.inputs.default_value = 0
    >>> at.inputs.transforms = ['ants_Warp.nii.gz', 'trans.mat']
    >>> at.inputs.invert_transform_flags = [False, True]
    >>> at.cmdline
    'antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 --input moving1.nii --interpolation Linear --output deformed_moving1.nii --reference-image fixed1.nii --transform ants_Warp.nii.gz --transform [ trans.mat, 1 ]'

    >>> at1 = ApplyTransforms()
    >>> at1.inputs.dimension = 3
    >>> at1.inputs.input_image = 'moving1.nii'
    >>> at1.inputs.reference_image = 'fixed1.nii'
    >>> at1.inputs.output_image = 'deformed_moving1.nii'
    >>> at1.inputs.interpolation = 'BSpline'
    >>> at1.inputs.interpolation_parameters = (5,)
    >>> at1.inputs.default_value = 0
    >>> at1.inputs.transforms = ['ants_Warp.nii.gz', 'trans.mat']
    >>> at1.inputs.invert_transform_flags = [False, False]
    >>> at1.cmdline
    'antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 --input moving1.nii --interpolation BSpline[ 5 ] --output deformed_moving1.nii --reference-image fixed1.nii --transform ants_Warp.nii.gz --transform trans.mat'

    Identity transforms may be used as part of a chain:

    >>> at2 = ApplyTransforms()
    >>> at2.inputs.dimension = 3
    >>> at2.inputs.input_image = 'moving1.nii'
    >>> at2.inputs.reference_image = 'fixed1.nii'
    >>> at2.inputs.output_image = 'deformed_moving1.nii'
    >>> at2.inputs.interpolation = 'BSpline'
    >>> at2.inputs.interpolation_parameters = (5,)
    >>> at2.inputs.default_value = 0
    >>> at2.inputs.transforms = ['identity', 'ants_Warp.nii.gz', 'trans.mat']
    >>> at2.cmdline
    'antsApplyTransforms --default-value 0 --dimensionality 3 --float 0 --input moving1.nii --interpolation BSpline[ 5 ] --output deformed_moving1.nii --reference-image fixed1.nii --transform identity --transform ants_Warp.nii.gz --transform trans.mat'
    """
    _cmd = 'antsApplyTransforms'
    input_spec = ApplyTransformsInputSpec
    output_spec = ApplyTransformsOutputSpec

    def _gen_filename(self, name):
        if name == 'output_image':
            output = self.inputs.output_image
            if not isdefined(output):
                _, name, ext = split_filename(self.inputs.input_image)
                output = name + self.inputs.out_postfix + ext
            return output
        return None

    def _get_transform_filenames(self):
        retval = []
        invert_flags = self.inputs.invert_transform_flags
        if not isdefined(invert_flags):
            invert_flags = [False] * len(self.inputs.transforms)
        elif len(self.inputs.transforms) != len(invert_flags):
            raise ValueError('ERROR: The invert_transform_flags list must have the same number of entries as the transforms list.')
        for transform, invert in zip(self.inputs.transforms, invert_flags):
            if invert:
                retval.append(f'--transform [ {transform}, 1 ]')
            else:
                retval.append(f'--transform {transform}')
        return ' '.join(retval)

    def _get_output_warped_filename(self):
        if isdefined(self.inputs.print_out_composite_warp_file):
            return '--output [ %s, %d ]' % (self._gen_filename('output_image'), int(self.inputs.print_out_composite_warp_file))
        else:
            return '--output %s' % self._gen_filename('output_image')

    def _format_arg(self, opt, spec, val):
        if opt == 'output_image':
            return self._get_output_warped_filename()
        elif opt == 'transforms':
            return self._get_transform_filenames()
        elif opt == 'interpolation':
            if self.inputs.interpolation in ['BSpline', 'MultiLabel', 'Gaussian'] and isdefined(self.inputs.interpolation_parameters):
                return '--interpolation %s[ %s ]' % (self.inputs.interpolation, ', '.join([str(param) for param in self.inputs.interpolation_parameters]))
            else:
                return '--interpolation %s' % self.inputs.interpolation
        return super(ApplyTransforms, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = os.path.abspath(self._gen_filename('output_image'))
        return outputs