import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class Qwarp(AFNICommand):
    """
    Allineate your images prior to passing them to this workflow.

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> qwarp = afni.Qwarp()
    >>> qwarp.inputs.in_file = 'sub-01_dir-LR_epi.nii.gz'
    >>> qwarp.inputs.nopadWARP = True
    >>> qwarp.inputs.base_file = 'sub-01_dir-RL_epi.nii.gz'
    >>> qwarp.inputs.plusminus = True
    >>> qwarp.cmdline
    '3dQwarp -base sub-01_dir-RL_epi.nii.gz -source sub-01_dir-LR_epi.nii.gz -nopadWARP -prefix ppp_sub-01_dir-LR_epi -plusminus'
    >>> res = qwarp.run()  # doctest: +SKIP

    >>> from nipype.interfaces import afni
    >>> qwarp = afni.Qwarp()
    >>> qwarp.inputs.in_file = 'structural.nii'
    >>> qwarp.inputs.base_file = 'mni.nii'
    >>> qwarp.inputs.resample = True
    >>> qwarp.cmdline
    '3dQwarp -base mni.nii -source structural.nii -prefix ppp_structural -resample'
    >>> res = qwarp.run()  # doctest: +SKIP

    >>> from nipype.interfaces import afni
    >>> qwarp = afni.Qwarp()
    >>> qwarp.inputs.in_file = 'structural.nii'
    >>> qwarp.inputs.base_file = 'epi.nii'
    >>> qwarp.inputs.out_file = 'anatSSQ.nii.gz'
    >>> qwarp.inputs.resample = True
    >>> qwarp.inputs.lpc = True
    >>> qwarp.inputs.verb = True
    >>> qwarp.inputs.iwarp = True
    >>> qwarp.inputs.blur = [0,3]
    >>> qwarp.cmdline
    '3dQwarp -base epi.nii -blur 0.0 3.0 -source structural.nii -iwarp -prefix anatSSQ.nii.gz -resample -verb -lpc'

    >>> res = qwarp.run()  # doctest: +SKIP

    >>> from nipype.interfaces import afni
    >>> qwarp = afni.Qwarp()
    >>> qwarp.inputs.in_file = 'structural.nii'
    >>> qwarp.inputs.base_file = 'mni.nii'
    >>> qwarp.inputs.duplo = True
    >>> qwarp.inputs.blur = [0,3]
    >>> qwarp.cmdline
    '3dQwarp -base mni.nii -blur 0.0 3.0 -duplo -source structural.nii -prefix ppp_structural'

    >>> res = qwarp.run()  # doctest: +SKIP

    >>> from nipype.interfaces import afni
    >>> qwarp = afni.Qwarp()
    >>> qwarp.inputs.in_file = 'structural.nii'
    >>> qwarp.inputs.base_file = 'mni.nii'
    >>> qwarp.inputs.duplo = True
    >>> qwarp.inputs.minpatch = 25
    >>> qwarp.inputs.blur = [0,3]
    >>> qwarp.inputs.out_file = 'Q25'
    >>> qwarp.cmdline
    '3dQwarp -base mni.nii -blur 0.0 3.0 -duplo -source structural.nii -minpatch 25 -prefix Q25'

    >>> res = qwarp.run()  # doctest: +SKIP
    >>> qwarp2 = afni.Qwarp()
    >>> qwarp2.inputs.in_file = 'structural.nii'
    >>> qwarp2.inputs.base_file = 'mni.nii'
    >>> qwarp2.inputs.blur = [0,2]
    >>> qwarp2.inputs.out_file = 'Q11'
    >>> qwarp2.inputs.inilev = 7
    >>> qwarp2.inputs.iniwarp = ['Q25_warp+tlrc.HEAD']
    >>> qwarp2.cmdline
    '3dQwarp -base mni.nii -blur 0.0 2.0 -source structural.nii -inilev 7 -iniwarp Q25_warp+tlrc.HEAD -prefix Q11'

    >>> res2 = qwarp2.run()  # doctest: +SKIP
    >>> res2 = qwarp2.run()  # doctest: +SKIP
    >>> qwarp3 = afni.Qwarp()
    >>> qwarp3.inputs.in_file = 'structural.nii'
    >>> qwarp3.inputs.base_file = 'mni.nii'
    >>> qwarp3.inputs.allineate = True
    >>> qwarp3.inputs.allineate_opts = '-cose lpa -verb'
    >>> qwarp3.cmdline
    "3dQwarp -allineate -allineate_opts '-cose lpa -verb' -base mni.nii -source structural.nii -prefix ppp_structural"

    >>> res3 = qwarp3.run()  # doctest: +SKIP

    See Also
    --------
    For complete details, see the `3dQwarp Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dQwarp.html>`__

    """
    _cmd = '3dQwarp'
    input_spec = QwarpInputSpec
    output_spec = QwarpOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'allineate_opts':
            return trait_spec.argstr % ("'" + value + "'")
        return super(Qwarp, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            prefix = self._gen_fname(self.inputs.in_file, suffix='_QW')
            outputtype = self.inputs.outputtype
            if outputtype == 'AFNI':
                ext = '.HEAD'
                suffix = '+tlrc'
            else:
                ext = Info.output_type_to_ext(outputtype)
                suffix = ''
        else:
            prefix = self.inputs.out_file
            ext_ind = max([prefix.lower().rfind('.nii.gz'), prefix.lower().rfind('.nii')])
            if ext_ind == -1:
                ext = '.HEAD'
                suffix = '+tlrc'
            else:
                ext = prefix[ext_ind:]
                suffix = ''
        out_dir = os.path.dirname(os.path.abspath(prefix))
        outputs['warped_source'] = fname_presuffix(prefix, suffix=suffix, use_ext=False, newpath=out_dir) + ext
        if not self.inputs.nowarp:
            outputs['source_warp'] = fname_presuffix(prefix, suffix='_WARP' + suffix, use_ext=False, newpath=out_dir) + ext
        if self.inputs.iwarp:
            outputs['base_warp'] = fname_presuffix(prefix, suffix='_WARPINV' + suffix, use_ext=False, newpath=out_dir) + ext
        if isdefined(self.inputs.out_weight_file):
            outputs['weights'] = os.path.abspath(self.inputs.out_weight_file)
        if self.inputs.plusminus:
            outputs['warped_source'] = fname_presuffix(prefix, suffix='_PLUS' + suffix, use_ext=False, newpath=out_dir) + ext
            outputs['warped_base'] = fname_presuffix(prefix, suffix='_MINUS' + suffix, use_ext=False, newpath=out_dir) + ext
            outputs['source_warp'] = fname_presuffix(prefix, suffix='_PLUS_WARP' + suffix, use_ext=False, newpath=out_dir) + ext
            outputs['base_warp'] = fname_presuffix(prefix, suffix='_MINUS_WARP' + suffix, use_ext=False, newpath=out_dir) + ext
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_fname(self.inputs.in_file, suffix='_QW')