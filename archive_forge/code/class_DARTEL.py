import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class DARTEL(SPMCommand):
    """Use spm DARTEL to create a template and flow fields

    http://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf#page=185

    Examples
    --------
    >>> import nipype.interfaces.spm as spm
    >>> dartel = spm.DARTEL()
    >>> dartel.inputs.image_files = [['rc1s1.nii','rc1s2.nii'],['rc2s1.nii', 'rc2s2.nii']]
    >>> dartel.run() # doctest: +SKIP

    """
    input_spec = DARTELInputSpec
    output_spec = DARTELOutputSpec
    _jobtype = 'tools'
    _jobname = 'dartel'

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt in ['image_files']:
            return scans_for_fnames(val, keep4d=True, separate_sessions=True)
        elif opt == 'regularization_form':
            mapper = {'Linear': 0, 'Membrane': 1, 'Bending': 2}
            return mapper[val]
        elif opt == 'iteration_parameters':
            params = []
            for param in val:
                new_param = {}
                new_param['its'] = param[0]
                new_param['rparam'] = list(param[1])
                new_param['K'] = param[2]
                new_param['slam'] = param[3]
                params.append(new_param)
            return params
        elif opt == 'optimization_parameters':
            new_param = {}
            new_param['lmreg'] = val[0]
            new_param['cyc'] = val[1]
            new_param['its'] = val[2]
            return [new_param]
        else:
            return super(DARTEL, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['template_files'] = []
        for i in range(6):
            outputs['template_files'].append(os.path.realpath('%s_%d.nii' % (self.inputs.template_prefix, i + 1)))
        outputs['final_template_file'] = os.path.realpath('%s_6.nii' % self.inputs.template_prefix)
        outputs['dartel_flow_fields'] = []
        for filename in self.inputs.image_files[0]:
            pth, base, ext = split_filename(filename)
            outputs['dartel_flow_fields'].append(os.path.realpath('u_%s_%s%s' % (base, self.inputs.template_prefix, ext)))
        return outputs