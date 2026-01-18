import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsExpand(FSSurfaceCommand):
    """
    Expands a surface (typically ?h.white) outwards while maintaining
    smoothness and self-intersection constraints.

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import MRIsExpand
    >>> mris_expand = MRIsExpand(thickness=True, distance=0.5)
    >>> mris_expand.inputs.in_file = 'lh.white'
    >>> mris_expand.cmdline
    'mris_expand -thickness lh.white 0.5 expanded'
    >>> mris_expand.inputs.out_name = 'graymid'
    >>> mris_expand.cmdline
    'mris_expand -thickness lh.white 0.5 graymid'
    """
    _cmd = 'mris_expand'
    input_spec = MRIsExpandInputSpec
    output_spec = MRIsExpandOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._associated_file(self.inputs.in_file, self.inputs.out_name)
        return outputs

    def normalize_filenames(self):
        """
        Filename normalization routine to perform only when run in Node
        context.
        Find full paths for pial, thickness and sphere files for copying.
        """
        in_file = self.inputs.in_file
        pial = self.inputs.pial
        if not isdefined(pial):
            pial = 'pial'
        self.inputs.pial = self._associated_file(in_file, pial)
        if isdefined(self.inputs.thickness) and self.inputs.thickness:
            thickness_name = self.inputs.thickness_name
            if not isdefined(thickness_name):
                thickness_name = 'thickness'
            self.inputs.thickness_name = self._associated_file(in_file, thickness_name)
        self.inputs.sphere = self._associated_file(in_file, self.inputs.sphere)