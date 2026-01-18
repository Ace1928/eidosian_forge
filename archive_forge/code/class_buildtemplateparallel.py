from builtins import range
import os
from glob import glob
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, OutputMultiPath
from ...utils.filemanip import split_filename
class buildtemplateparallel(ANTSCommand):
    """Generate a optimal average template

    .. warning::

      This can take a VERY long time to complete

    Examples
    --------

    >>> from nipype.interfaces.ants.legacy import buildtemplateparallel
    >>> tmpl = buildtemplateparallel()
    >>> tmpl.inputs.in_files = ['T1.nii', 'structural.nii']
    >>> tmpl.inputs.max_iterations = [30, 90, 20]
    >>> tmpl.cmdline
    'buildtemplateparallel.sh -d 3 -i 4 -m 30x90x20 -o antsTMPL_ -c 0 -t GR T1.nii structural.nii'

    """
    _cmd = 'buildtemplateparallel.sh'
    input_spec = buildtemplateparallelInputSpec
    output_spec = buildtemplateparallelOutputSpec

    def _format_arg(self, opt, spec, val):
        if opt == 'num_cores':
            if self.inputs.parallelization == 2:
                return '-j ' + str(val)
            else:
                return ''
        if opt == 'in_files':
            if self.inputs.use_first_as_target:
                start = '-z '
            else:
                start = ''
            return start + ' '.join((name for name in val))
        return super(buildtemplateparallel, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['template_files'] = []
        for i in range(len(glob(os.path.realpath('*iteration*')))):
            temp = os.path.realpath('%s_iteration_%d/%stemplate.nii.gz' % (self.inputs.transformation_model, i, self.inputs.out_prefix))
            os.rename(temp, os.path.realpath('%s_iteration_%d/%stemplate_i%d.nii.gz' % (self.inputs.transformation_model, i, self.inputs.out_prefix, i)))
            file_ = '%s_iteration_%d/%stemplate_i%d.nii.gz' % (self.inputs.transformation_model, i, self.inputs.out_prefix, i)
            outputs['template_files'].append(os.path.realpath(file_))
            outputs['final_template_file'] = os.path.realpath('%stemplate.nii.gz' % self.inputs.out_prefix)
        outputs['subject_outfiles'] = []
        for filename in self.inputs.in_files:
            _, base, _ = split_filename(filename)
            temp = glob(os.path.realpath('%s%s*' % (self.inputs.out_prefix, base)))
            for file_ in temp:
                outputs['subject_outfiles'].append(file_)
        return outputs