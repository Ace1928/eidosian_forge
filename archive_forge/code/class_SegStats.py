import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class SegStats(FSCommand):
    """Use FreeSurfer mri_segstats for ROI analysis

    Examples
    --------
    >>> import nipype.interfaces.freesurfer as fs
    >>> ss = fs.SegStats()
    >>> ss.inputs.annot = ('PWS04', 'lh', 'aparc')
    >>> ss.inputs.in_file = 'functional.nii'
    >>> ss.inputs.subjects_dir = '.'
    >>> ss.inputs.avgwf_txt_file = 'avgwf.txt'
    >>> ss.inputs.summary_file = 'summary.stats'
    >>> ss.cmdline
    'mri_segstats --annot PWS04 lh aparc --avgwf ./avgwf.txt --i functional.nii --sum ./summary.stats'

    """
    _cmd = 'mri_segstats'
    input_spec = SegStatsInputSpec
    output_spec = SegStatsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.summary_file):
            outputs['summary_file'] = os.path.abspath(self.inputs.summary_file)
        else:
            outputs['summary_file'] = os.path.join(os.getcwd(), 'summary.stats')
        suffices = dict(avgwf_txt_file='_avgwf.txt', avgwf_file='_avgwf.nii.gz', sf_avg_file='sfavg.txt')
        if isdefined(self.inputs.segmentation_file):
            _, src = os.path.split(self.inputs.segmentation_file)
        if isdefined(self.inputs.annot):
            src = '_'.join(self.inputs.annot)
        if isdefined(self.inputs.surf_label):
            src = '_'.join(self.inputs.surf_label)
        for name, suffix in list(suffices.items()):
            value = getattr(self.inputs, name)
            if isdefined(value):
                if isinstance(value, bool):
                    outputs[name] = fname_presuffix(src, suffix=suffix, newpath=os.getcwd(), use_ext=False)
                else:
                    outputs[name] = os.path.abspath(value)
        return outputs

    def _format_arg(self, name, spec, value):
        if name in ('summary_file', 'avgwf_txt_file'):
            if not isinstance(value, bool):
                if not os.path.isabs(value):
                    value = os.path.join('.', value)
        if name in ['avgwf_txt_file', 'avgwf_file', 'sf_avg_file']:
            if isinstance(value, bool):
                fname = self._list_outputs()[name]
            else:
                fname = value
            return spec.argstr % fname
        elif name == 'in_intensity':
            intensity_name = os.path.basename(self.inputs.in_intensity).replace('.mgz', '')
            return spec.argstr % (value, intensity_name)
        return super(SegStats, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):
        if name == 'summary_file':
            return self._list_outputs()[name]
        return None