import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class MCFLIRT(FSLCommand):
    """FSL MCFLIRT wrapper for within-modality motion correction

    For complete details, see the `MCFLIRT Documentation.
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MCFLIRT>`_

    Examples
    --------
    >>> from nipype.interfaces import fsl
    >>> mcflt = fsl.MCFLIRT()
    >>> mcflt.inputs.in_file = 'functional.nii'
    >>> mcflt.inputs.cost = 'mutualinfo'
    >>> mcflt.inputs.out_file = 'moco.nii'
    >>> mcflt.cmdline
    'mcflirt -in functional.nii -cost mutualinfo -out moco.nii'
    >>> res = mcflt.run()  # doctest: +SKIP

    """
    _cmd = 'mcflirt'
    input_spec = MCFLIRTInputSpec
    output_spec = MCFLIRTOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'interpolation':
            if value == 'trilinear':
                return ''
            else:
                return spec.argstr % value
        return super(MCFLIRT, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_outfilename()
        output_dir = os.path.dirname(outputs['out_file'])
        if isdefined(self.inputs.stats_imgs) and self.inputs.stats_imgs:
            if LooseVersion(Info.version()) < LooseVersion('6.0.0'):
                outputs['variance_img'] = self._gen_fname(outputs['out_file'] + '_variance.ext', cwd=output_dir)
                outputs['std_img'] = self._gen_fname(outputs['out_file'] + '_sigma.ext', cwd=output_dir)
            else:
                outputs['variance_img'] = self._gen_fname(outputs['out_file'], suffix='_variance', cwd=output_dir)
                outputs['std_img'] = self._gen_fname(outputs['out_file'], suffix='_sigma', cwd=output_dir)
        if isdefined(self.inputs.mean_vol) and self.inputs.mean_vol:
            if LooseVersion(Info.version()) < LooseVersion('6.0.0'):
                outputs['mean_img'] = self._gen_fname(outputs['out_file'] + '_mean_reg.ext', cwd=output_dir)
            else:
                outputs['mean_img'] = self._gen_fname(outputs['out_file'], suffix='_mean_reg', cwd=output_dir)
        if isdefined(self.inputs.save_mats) and self.inputs.save_mats:
            _, filename = os.path.split(outputs['out_file'])
            matpathname = os.path.join(output_dir, filename + '.mat')
            _, _, _, timepoints = load(self.inputs.in_file).shape
            outputs['mat_file'] = []
            for t in range(timepoints):
                outputs['mat_file'].append(os.path.join(matpathname, 'MAT_%04d' % t))
        if isdefined(self.inputs.save_plots) and self.inputs.save_plots:
            outputs['par_file'] = outputs['out_file'] + '.par'
        if isdefined(self.inputs.save_rms) and self.inputs.save_rms:
            outfile = outputs['out_file']
            outputs['rms_files'] = [outfile + '_abs.rms', outfile + '_rel.rms']
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        return None

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if isdefined(out_file):
            out_file = os.path.realpath(out_file)
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            out_file = self._gen_fname(self.inputs.in_file, suffix='_mcf')
        return os.path.abspath(out_file)