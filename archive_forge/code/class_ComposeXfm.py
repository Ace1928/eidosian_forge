from ..base import TraitedSpec, CommandLineInputSpec, traits, File, isdefined
from ...utils.filemanip import fname_presuffix, split_filename
from .base import CommandLineDtitk, DTITKRenameMixin
import os
class ComposeXfm(CommandLineDtitk):
    """
     Combines diffeomorphic and affine transforms

    Example
    -------

    >>> from nipype.interfaces import dtitk
    >>> node = dtitk.ComposeXfm()
    >>> node.inputs.in_df = 'im_warp.df.nii'
    >>> node.inputs.in_aff= 'im_affine.aff'
    >>> node.cmdline
    'dfRightComposeAffine -aff im_affine.aff -df im_warp.df.nii -out
     im_warp_affdf.df.nii'
    >>> node.run() # doctest: +SKIP
    """
    input_spec = ComposeXfmInputSpec
    output_spec = ComposeXfmOutputSpec
    _cmd = 'dfRightComposeAffine'

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_file = self.inputs.out_file
        if not isdefined(out_file):
            out_file = self._gen_filename('out_file')
        outputs['out_file'] = os.path.abspath(out_file)
        return outputs

    def _gen_filename(self, name):
        if name != 'out_file':
            return
        path, base, ext = split_filename(self.inputs.in_df)
        suffix = '_affdf'
        if base.endswith('.df'):
            suffix += '.df'
            base = base[:-3]
        return fname_presuffix(base, suffix=suffix + ext, use_ext=False)