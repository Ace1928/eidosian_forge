import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class MPRtoMNI305(FSScriptCommand):
    """
    For complete details, see FreeSurfer documentation

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import MPRtoMNI305, Info
    >>> mprtomni305 = MPRtoMNI305()
    >>> mprtomni305.inputs.target = 'structural.nii'
    >>> mprtomni305.inputs.reference_dir = '.' # doctest: +SKIP
    >>> mprtomni305.cmdline # doctest: +SKIP
    'mpr2mni305 output'
    >>> mprtomni305.inputs.out_file = 'struct_out' # doctest: +SKIP
    >>> mprtomni305.cmdline # doctest: +SKIP
    'mpr2mni305 struct_out' # doctest: +SKIP
    >>> mprtomni305.inputs.environ['REFDIR'] == os.path.join(Info.home(), 'average') # doctest: +SKIP
    True
    >>> mprtomni305.inputs.environ['MPR2MNI305_TARGET'] # doctest: +SKIP
    'structural'
    >>> mprtomni305.run() # doctest: +SKIP

    """
    _cmd = 'mpr2mni305'
    input_spec = MPRtoMNI305InputSpec
    output_spec = MPRtoMNI305OutputSpec

    def __init__(self, **inputs):
        super(MPRtoMNI305, self).__init__(**inputs)
        self.inputs.on_trait_change(self._environ_update, 'target')
        self.inputs.on_trait_change(self._environ_update, 'reference_dir')

    def _format_arg(self, opt, spec, val):
        if opt in ['target', 'reference_dir']:
            return ''
        elif opt == 'in_file':
            _, retval, ext = split_filename(val)
            copyfile(val, os.path.abspath(retval + ext), copy=True, hashmethod='content')
            return retval
        return super(MPRtoMNI305, self)._format_arg(opt, spec, val)

    def _environ_update(self):
        refdir = self.inputs.reference_dir
        target = self.inputs.target
        self.inputs.environ['MPR2MNI305_TARGET'] = target
        self.inputs.environ['REFDIR'] = refdir

    def _get_fname(self, fname):
        return split_filename(fname)[1]

    def _list_outputs(self):
        outputs = super(MPRtoMNI305, self)._list_outputs()
        fullname = '_'.join([self._get_fname(self.inputs.in_file), 'to', self.inputs.target, 't4', 'vox2vox.txt'])
        outputs['out_file'] = os.path.abspath(fullname)
        return outputs