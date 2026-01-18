import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class BET(FSLCommand):
    """FSL BET wrapper for skull stripping

    For complete details, see the `BET Documentation.
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide>`_

    Examples
    --------
    >>> from nipype.interfaces import fsl
    >>> btr = fsl.BET()
    >>> btr.inputs.in_file = 'structural.nii'
    >>> btr.inputs.frac = 0.7
    >>> btr.inputs.out_file = 'brain_anat.nii'
    >>> btr.cmdline
    'bet structural.nii brain_anat.nii -f 0.70'
    >>> res = btr.run() # doctest: +SKIP

    """
    _cmd = 'bet'
    input_spec = BETInputSpec
    output_spec = BETOutputSpec

    def _run_interface(self, runtime):
        runtime = super(BET, self)._run_interface(runtime)
        if runtime.stderr:
            self.raise_exception(runtime)
        return runtime

    def _format_arg(self, name, spec, value):
        formatted = super(BET, self)._format_arg(name, spec, value)
        if name == 'in_file':
            return op.relpath(formatted, start=os.getcwd())
        return formatted

    def _gen_outfilename(self):
        out_file = self.inputs.out_file
        if not isdefined(out_file) and isdefined(self.inputs.in_file):
            out_file = self._gen_fname(self.inputs.in_file, suffix='_brain')
            return op.relpath(out_file, start=os.getcwd())
        return out_file

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_outfilename())
        basename = os.path.basename(outputs['out_file'])
        cwd = os.path.dirname(outputs['out_file'])
        kwargs = {'basename': basename, 'cwd': cwd}
        if isdefined(self.inputs.mesh) and self.inputs.mesh or (isdefined(self.inputs.surfaces) and self.inputs.surfaces):
            outputs['meshfile'] = self._gen_fname(suffix='_mesh.vtk', change_ext=False, **kwargs)
        if isdefined(self.inputs.mask) and self.inputs.mask or (isdefined(self.inputs.reduce_bias) and self.inputs.reduce_bias):
            outputs['mask_file'] = self._gen_fname(suffix='_mask', **kwargs)
        if isdefined(self.inputs.outline) and self.inputs.outline:
            outputs['outline_file'] = self._gen_fname(suffix='_overlay', **kwargs)
        if isdefined(self.inputs.surfaces) and self.inputs.surfaces:
            outputs['inskull_mask_file'] = self._gen_fname(suffix='_inskull_mask', **kwargs)
            outputs['inskull_mesh_file'] = self._gen_fname(suffix='_inskull_mesh', **kwargs)
            outputs['outskull_mask_file'] = self._gen_fname(suffix='_outskull_mask', **kwargs)
            outputs['outskull_mesh_file'] = self._gen_fname(suffix='_outskull_mesh', **kwargs)
            outputs['outskin_mask_file'] = self._gen_fname(suffix='_outskin_mask', **kwargs)
            outputs['outskin_mesh_file'] = self._gen_fname(suffix='_outskin_mesh', **kwargs)
            outputs['skull_mask_file'] = self._gen_fname(suffix='_skull_mask', **kwargs)
        if isdefined(self.inputs.skull) and self.inputs.skull:
            outputs['skull_file'] = self._gen_fname(suffix='_skull', **kwargs)
        if isdefined(self.inputs.no_output) and self.inputs.no_output:
            outputs['out_file'] = Undefined
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        return None