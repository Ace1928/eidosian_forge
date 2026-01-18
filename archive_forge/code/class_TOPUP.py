import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class TOPUP(FSLCommand):
    """
    Interface for FSL topup, a tool for estimating and correcting
    susceptibility induced distortions. See FSL documentation for
    `reference <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP>`_,
    `usage examples
    <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/ExampleTopupFollowedByApplytopup>`_,
    and `exemplary config files
    <https://github.com/ahheckel/FSL-scripts/blob/master/rsc/fsl/fsl4/topup/b02b0.cnf>`_.

    Examples
    --------

    >>> from nipype.interfaces.fsl import TOPUP
    >>> topup = TOPUP()
    >>> topup.inputs.in_file = "b0_b0rev.nii"
    >>> topup.inputs.encoding_file = "topup_encoding.txt"
    >>> topup.inputs.output_type = "NIFTI_GZ"
    >>> topup.cmdline # doctest: +ELLIPSIS
    'topup --config=b02b0.cnf --datain=topup_encoding.txt --imain=b0_b0rev.nii --out=b0_b0rev_base --iout=b0_b0rev_corrected.nii.gz --fout=b0_b0rev_field.nii.gz --jacout=jac --logout=b0_b0rev_topup.log --rbmout=xfm --dfout=warpfield'
    >>> res = topup.run() # doctest: +SKIP

    """
    _cmd = 'topup'
    input_spec = TOPUPInputSpec
    output_spec = TOPUPOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'encoding_direction':
            return trait_spec.argstr % self._generate_encfile()
        if name == 'out_base':
            path, name, ext = split_filename(value)
            if path != '':
                if not os.path.exists(path):
                    raise ValueError('out_base path must exist if provided')
        return super(TOPUP, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = super(TOPUP, self)._list_outputs()
        del outputs['out_base']
        base_path = None
        if isdefined(self.inputs.out_base):
            base_path, base, _ = split_filename(self.inputs.out_base)
            if base_path == '':
                base_path = None
        else:
            base = split_filename(self.inputs.in_file)[1] + '_base'
        outputs['out_fieldcoef'] = self._gen_fname(base, suffix='_fieldcoef', cwd=base_path)
        outputs['out_movpar'] = self._gen_fname(base, suffix='_movpar', ext='.txt', cwd=base_path)
        n_vols = nb.load(self.inputs.in_file).shape[-1]
        ext = Info.output_type_to_ext(self.inputs.output_type)
        fmt = os.path.abspath('{prefix}_{i:02d}{ext}').format
        outputs['out_warps'] = [fmt(prefix=self.inputs.out_warp_prefix, i=i, ext=ext) for i in range(1, n_vols + 1)]
        outputs['out_jacs'] = [fmt(prefix=self.inputs.out_jac_prefix, i=i, ext=ext) for i in range(1, n_vols + 1)]
        outputs['out_mats'] = [fmt(prefix=self.inputs.out_mat_prefix, i=i, ext='.mat') for i in range(1, n_vols + 1)]
        if isdefined(self.inputs.encoding_direction):
            outputs['out_enc_file'] = self._get_encfilename()
        return outputs

    def _get_encfilename(self):
        out_file = os.path.join(os.getcwd(), '%s_encfile.txt' % split_filename(self.inputs.in_file)[1])
        return out_file

    def _generate_encfile(self):
        """Generate a topup compatible encoding file based on given directions"""
        out_file = self._get_encfilename()
        durations = self.inputs.readout_times
        if len(self.inputs.encoding_direction) != len(durations):
            if len(self.inputs.readout_times) != 1:
                raise ValueError('Readout time must be a float or match thelength of encoding directions')
            durations = durations * len(self.inputs.encoding_direction)
        lines = []
        for idx, encdir in enumerate(self.inputs.encoding_direction):
            direction = 1.0
            if encdir.endswith('-'):
                direction = -1.0
            line = [float(val[0] == encdir[0]) * direction for val in ['x', 'y', 'z']] + [durations[idx]]
            lines.append(line)
        np.savetxt(out_file, np.array(lines), fmt=b'%d %d %d %.8f')
        return out_file

    def _overload_extension(self, value, name=None):
        if name == 'out_base':
            return value
        return super(TOPUP, self)._overload_extension(value, name)