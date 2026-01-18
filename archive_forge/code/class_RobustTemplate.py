import os
from ... import logging
from ..base import TraitedSpec, File, traits, InputMultiPath, OutputMultiPath, isdefined
from .base import FSCommand, FSTraitedSpec, FSCommandOpenMP, FSTraitedSpecOpenMP
class RobustTemplate(FSCommandOpenMP):
    """construct an unbiased robust template for longitudinal volumes

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import RobustTemplate
    >>> template = RobustTemplate()
    >>> template.inputs.in_files = ['structural.nii', 'functional.nii']
    >>> template.inputs.auto_detect_sensitivity = True
    >>> template.inputs.average_metric = 'mean'
    >>> template.inputs.initial_timepoint = 1
    >>> template.inputs.fixed_timepoint = True
    >>> template.inputs.no_iteration = True
    >>> template.inputs.subsample_threshold = 200
    >>> template.cmdline  #doctest:
    'mri_robust_template --satit --average 0 --fixtp --mov structural.nii functional.nii --inittp 1 --noit --template mri_robust_template_out.mgz --subsample 200'
    >>> template.inputs.out_file = 'T1.nii'
    >>> template.cmdline  #doctest:
    'mri_robust_template --satit --average 0 --fixtp --mov structural.nii functional.nii --inittp 1 --noit --template T1.nii --subsample 200'

    >>> template.inputs.transform_outputs = ['structural.lta',
    ...                                      'functional.lta']
    >>> template.inputs.scaled_intensity_outputs = ['structural-iscale.txt',
    ...                                             'functional-iscale.txt']
    >>> template.cmdline    #doctest: +ELLIPSIS
    'mri_robust_template --satit --average 0 --fixtp --mov structural.nii functional.nii --inittp 1 --noit --template T1.nii --iscaleout .../structural-iscale.txt .../functional-iscale.txt --subsample 200 --lta .../structural.lta .../functional.lta'

    >>> template.inputs.transform_outputs = True
    >>> template.inputs.scaled_intensity_outputs = True
    >>> template.cmdline    #doctest: +ELLIPSIS
    'mri_robust_template --satit --average 0 --fixtp --mov structural.nii functional.nii --inittp 1 --noit --template T1.nii --iscaleout .../is1.txt .../is2.txt --subsample 200 --lta .../tp1.lta .../tp2.lta'

    >>> template.run()  #doctest: +SKIP

    References
    ----------
    [https://surfer.nmr.mgh.harvard.edu/fswiki/mri_robust_template]

    """
    _cmd = 'mri_robust_template'
    input_spec = RobustTemplateInputSpec
    output_spec = RobustTemplateOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'average_metric':
            return spec.argstr % {'mean': 0, 'median': 1}[value]
        if name in ('transform_outputs', 'scaled_intensity_outputs'):
            value = self._list_outputs()[name]
        return super(RobustTemplate, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        n_files = len(self.inputs.in_files)
        fmt = '{}{:02d}.{}' if n_files > 9 else '{}{:d}.{}'
        if isdefined(self.inputs.transform_outputs):
            fnames = self.inputs.transform_outputs
            if fnames is True:
                fnames = [fmt.format('tp', i + 1, 'lta') for i in range(n_files)]
            outputs['transform_outputs'] = [os.path.abspath(x) for x in fnames]
        if isdefined(self.inputs.scaled_intensity_outputs):
            fnames = self.inputs.scaled_intensity_outputs
            if fnames is True:
                fnames = [fmt.format('is', i + 1, 'txt') for i in range(n_files)]
            outputs['scaled_intensity_outputs'] = [os.path.abspath(x) for x in fnames]
        return outputs