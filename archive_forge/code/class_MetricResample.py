import os
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import WBCommand
from ... import logging
class MetricResample(WBCommand):
    """
    Resample a metric file to a different mesh

    Resamples a metric file, given two spherical surfaces that are in
    register.  If ``ADAP_BARY_AREA`` is used, exactly one of -area-surfs or
    ``-area-metrics`` must be specified.

    The ``ADAP_BARY_AREA`` method is recommended for ordinary metric data,
    because it should use all data while downsampling, unlike ``BARYCENTRIC``.
    The recommended areas option for most data is individual midthicknesses
    for individual data, and averaged vertex area metrics from individual
    midthicknesses for group average data.

    The ``-current-roi`` option only masks the input, the output may be slightly
    dilated in comparison, consider using ``-metric-mask`` on the output when
    using ``-current-roi``.

    The ``-largest option`` results in nearest vertex behavior when used with
    ``BARYCENTRIC``.  When resampling a binary metric, consider thresholding at
    0.5 after resampling rather than using ``-largest``.

    >>> from nipype.interfaces.workbench import MetricResample
    >>> metres = MetricResample()
    >>> metres.inputs.in_file = 'sub-01_task-rest_bold_space-fsaverage5.L.func.gii'
    >>> metres.inputs.method = 'ADAP_BARY_AREA'
    >>> metres.inputs.current_sphere = 'fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii'
    >>> metres.inputs.new_sphere = 'fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii'
    >>> metres.inputs.area_metrics = True
    >>> metres.inputs.current_area = 'fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii'
    >>> metres.inputs.new_area = 'fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii'
    >>> metres.cmdline
    'wb_command -metric-resample sub-01_task-rest_bold_space-fsaverage5.L.func.gii     fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii     fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii     ADAP_BARY_AREA fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.out     -area-metrics fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii     fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii'
    """
    input_spec = MetricResampleInputSpec
    output_spec = MetricResampleOutputSpec
    _cmd = 'wb_command -metric-resample'

    def _format_arg(self, opt, spec, val):
        if opt in ['current_area', 'new_area']:
            if not self.inputs.area_surfs and (not self.inputs.area_metrics):
                raise ValueError('{} was set but neither area_surfs or area_metrics were set'.format(opt))
        if opt == 'method':
            if val == 'ADAP_BARY_AREA' and (not self.inputs.area_surfs) and (not self.inputs.area_metrics):
                raise ValueError('Exactly one of area_surfs or area_metrics must be specified')
        if opt == 'valid_roi_out' and val:
            roi_out = self._gen_filename(self.inputs.in_file, suffix='_roi')
            iflogger.info('Setting roi output file as', roi_out)
            spec.argstr += ' ' + roi_out
        return super(MetricResample, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = super(MetricResample, self)._list_outputs()
        if self.inputs.valid_roi_out:
            roi_file = self._gen_filename(self.inputs.in_file, suffix='_roi')
            outputs['roi_file'] = os.path.abspath(roi_file)
        return outputs