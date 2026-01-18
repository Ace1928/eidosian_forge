from ... import logging, LooseVersion
from ...utils.filemanip import which
from ..base import (
class MRTrix3Base(CommandLine):

    def _format_arg(self, name, trait_spec, value):
        if name == 'nthreads' and value == 0:
            value = 1
            try:
                from multiprocessing import cpu_count
                value = cpu_count()
            except:
                iflogger.warning('Number of threads could not be computed')
                pass
            return trait_spec.argstr % value
        if name == 'in_bvec':
            return trait_spec.argstr % (value, self.inputs.in_bval)
        if name == 'out_bvec':
            return trait_spec.argstr % (value, self.inputs.out_bval)
        return super(MRTrix3Base, self)._format_arg(name, trait_spec, value)

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        try:
            if isdefined(self.inputs.grad_file) or isdefined(self.inputs.grad_fsl):
                skip += ['in_bvec', 'in_bval']
            is_bvec = isdefined(self.inputs.in_bvec)
            is_bval = isdefined(self.inputs.in_bval)
            if is_bvec or is_bval:
                if not is_bvec or not is_bval:
                    raise RuntimeError('If using bvecs and bvals inputs, bothshould be defined')
                skip += ['in_bval']
        except AttributeError:
            pass
        return super(MRTrix3Base, self)._parse_inputs(skip=skip)

    @property
    def version(self):
        return Info.version()