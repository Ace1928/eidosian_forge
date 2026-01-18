import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class EulerNumber(FSCommand):
    """
    This program computes EulerNumber for a cortical surface

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import EulerNumber
    >>> ft = EulerNumber()
    >>> ft.inputs.in_file = 'lh.pial'
    >>> ft.cmdline
    'mris_euler_number lh.pial'
    """
    _cmd = 'mris_euler_number'
    input_spec = EulerNumberInputSpec
    output_spec = EulerNumberOutputSpec

    def _run_interface(self, runtime):
        runtime = super()._run_interface(runtime)
        self._parse_output(runtime.stdout, runtime.stderr)
        return runtime

    def _parse_output(self, stdout, stderr):
        """Parse stdout / stderr and extract defects"""
        m = re.search('(?<=total defect index = )\\d+', stdout or stderr)
        if m is None:
            raise RuntimeError('Could not fetch defect index')
        self._defects = int(m.group())

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['defects'] = self._defects
        outputs['euler'] = 2 - 2 * self._defects
        return outputs