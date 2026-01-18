import os
import re
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
class DTIRecon(CommandLine):
    """Use dti_recon to generate tensors and other maps"""
    input_spec = DTIReconInputSpec
    output_spec = DTIReconOutputSpec
    _cmd = 'dti_recon'

    def _create_gradient_matrix(self, bvecs_file, bvals_file):
        _gradient_matrix_file = 'gradient_matrix.txt'
        with open(bvals_file) as fbvals:
            bvals = [val for val in re.split('\\s+', fbvals.readline().strip())]
        with open(bvecs_file) as fbvecs:
            bvecs_x = fbvecs.readline().split()
            bvecs_y = fbvecs.readline().split()
            bvecs_z = fbvecs.readline().split()
        with open(_gradient_matrix_file, 'w') as gradient_matrix_f:
            for i in range(len(bvals)):
                gradient_matrix_f.write('%s, %s, %s, %s\n' % (bvecs_x[i], bvecs_y[i], bvecs_z[i], bvals[i]))
        return _gradient_matrix_file

    def _format_arg(self, name, spec, value):
        if name == 'bvecs':
            new_val = self._create_gradient_matrix(self.inputs.bvecs, self.inputs.bvals)
            return super(DTIRecon, self)._format_arg('bvecs', spec, new_val)
        return super(DTIRecon, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        out_prefix = self.inputs.out_prefix
        output_type = self.inputs.output_type
        outputs = self.output_spec().get()
        outputs['ADC'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_adc.' + output_type))
        outputs['B0'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_b0.' + output_type))
        outputs['L1'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_e1.' + output_type))
        outputs['L2'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_e2.' + output_type))
        outputs['L3'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_e3.' + output_type))
        outputs['exp'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_exp.' + output_type))
        outputs['FA'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_fa.' + output_type))
        outputs['FA_color'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_fa_color.' + output_type))
        outputs['tensor'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_tensor.' + output_type))
        outputs['V1'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_v1.' + output_type))
        outputs['V2'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_v2.' + output_type))
        outputs['V3'] = os.path.abspath(fname_presuffix('', prefix=out_prefix, suffix='_v3.' + output_type))
        return outputs