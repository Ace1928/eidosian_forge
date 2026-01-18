import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class Paint(FSCommand):
    """
    This program is useful for extracting one of the arrays ("a variable")
    from a surface-registration template file. The output is a file
    containing a surface-worth of per-vertex values, saved in "curvature"
    format. Because the template data is sampled to a particular surface
    mesh, this conjures the idea of "painting to a surface".

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import Paint
    >>> paint = Paint()
    >>> paint.inputs.in_surf = 'lh.pial'
    >>> paint.inputs.template = 'aseg.mgz'
    >>> paint.inputs.averages = 5
    >>> paint.inputs.out_file = 'lh.avg_curv'
    >>> paint.cmdline
    'mrisp_paint -a 5 aseg.mgz lh.pial lh.avg_curv'
    """
    _cmd = 'mrisp_paint'
    input_spec = PaintInputSpec
    output_spec = PaintOutputSpec

    def _format_arg(self, opt, spec, val):
        if opt == 'template':
            if isdefined(self.inputs.template_param):
                return spec.argstr % (val + '#' + str(self.inputs.template_param))
        return super(Paint, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs