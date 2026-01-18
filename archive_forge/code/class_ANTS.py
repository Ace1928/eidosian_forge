import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
class ANTS(ANTSCommand):
    """ANTS wrapper for registration of images
    (old, use Registration instead)

    Examples
    --------

    >>> from nipype.interfaces.ants import ANTS
    >>> ants = ANTS()
    >>> ants.inputs.dimension = 3
    >>> ants.inputs.output_transform_prefix = 'MY'
    >>> ants.inputs.metric = ['CC']
    >>> ants.inputs.fixed_image = ['T1.nii']
    >>> ants.inputs.moving_image = ['resting.nii']
    >>> ants.inputs.metric_weight = [1.0]
    >>> ants.inputs.radius = [5]
    >>> ants.inputs.transformation_model = 'SyN'
    >>> ants.inputs.gradient_step_length = 0.25
    >>> ants.inputs.number_of_iterations = [50, 35, 15]
    >>> ants.inputs.use_histogram_matching = True
    >>> ants.inputs.mi_option = [32, 16000]
    >>> ants.inputs.regularization = 'Gauss'
    >>> ants.inputs.regularization_gradient_field_sigma = 3
    >>> ants.inputs.regularization_deformation_field_sigma = 0
    >>> ants.inputs.number_of_affine_iterations = [10000,10000,10000,10000,10000]
    >>> ants.cmdline
    'ANTS 3 --MI-option 32x16000 --image-metric CC[ T1.nii, resting.nii, 1, 5 ] --number-of-affine-iterations 10000x10000x10000x10000x10000 --number-of-iterations 50x35x15 --output-naming MY --regularization Gauss[3.0,0.0] --transformation-model SyN[0.25] --use-Histogram-Matching 1'
    """
    _cmd = 'ANTS'
    input_spec = ANTSInputSpec
    output_spec = ANTSOutputSpec

    def _image_metric_constructor(self):
        retval = []
        intensity_based = ['CC', 'MI', 'SMI', 'PR', 'SSD', 'MSQ']
        point_set_based = ['PSE', 'JTB']
        for ii in range(len(self.inputs.moving_image)):
            if self.inputs.metric[ii] in intensity_based:
                retval.append('--image-metric %s[ %s, %s, %g, %d ]' % (self.inputs.metric[ii], self.inputs.fixed_image[ii], self.inputs.moving_image[ii], self.inputs.metric_weight[ii], self.inputs.radius[ii]))
            elif self.inputs.metric[ii] == point_set_based:
                pass
        return ' '.join(retval)

    def _transformation_constructor(self):
        model = self.inputs.transformation_model
        step_length = self.inputs.gradient_step_length
        time_step = self.inputs.number_of_time_steps
        delta_time = self.inputs.delta_time
        symmetry_type = self.inputs.symmetry_type
        retval = ['--transformation-model %s' % model]
        parameters = []
        for elem in (step_length, time_step, delta_time, symmetry_type):
            if elem is not traits.Undefined:
                parameters.append('%#.2g' % elem)
        if len(parameters) > 0:
            if len(parameters) > 1:
                parameters = ','.join(parameters)
            else:
                parameters = ''.join(parameters)
            retval.append('[%s]' % parameters)
        return ''.join(retval)

    def _regularization_constructor(self):
        return '--regularization {0}[{1},{2}]'.format(self.inputs.regularization, self.inputs.regularization_gradient_field_sigma, self.inputs.regularization_deformation_field_sigma)

    def _affine_gradient_descent_option_constructor(self):
        values = self.inputs.affine_gradient_descent_option
        defaults = [0.1, 0.5, 0.0001, 0.0001]
        for ii in range(len(defaults)):
            try:
                defaults[ii] = values[ii]
            except IndexError:
                break
        parameters = self._format_xarray(['%g' % defaults[index] for index in range(4)])
        retval = ['--affine-gradient-descent-option', parameters]
        return ' '.join(retval)

    def _format_arg(self, opt, spec, val):
        if opt == 'moving_image':
            return self._image_metric_constructor()
        elif opt == 'transformation_model':
            return self._transformation_constructor()
        elif opt == 'regularization':
            return self._regularization_constructor()
        elif opt == 'affine_gradient_descent_option':
            return self._affine_gradient_descent_option_constructor()
        elif opt == 'use_histogram_matching':
            if self.inputs.use_histogram_matching:
                return '--use-Histogram-Matching 1'
            else:
                return '--use-Histogram-Matching 0'
        return super(ANTS, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['affine_transform'] = os.path.abspath(self.inputs.output_transform_prefix + 'Affine.txt')
        outputs['warp_transform'] = os.path.abspath(self.inputs.output_transform_prefix + 'Warp.nii.gz')
        outputs['inverse_warp_transform'] = os.path.abspath(self.inputs.output_transform_prefix + 'InverseWarp.nii.gz')
        return outputs