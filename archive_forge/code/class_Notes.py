import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Notes(CommandLine):
    """A program to add, delete, and show notes for AFNI datasets.

    For complete details, see the `3dNotes Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dNotes.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> notes = afni.Notes()
    >>> notes.inputs.in_file = 'functional.HEAD'
    >>> notes.inputs.add = 'This note is added.'
    >>> notes.inputs.add_history = 'This note is added to history.'
    >>> notes.cmdline
    '3dNotes -a "This note is added." -h "This note is added to history." functional.HEAD'
    >>> res = notes.run()  # doctest: +SKIP

    """
    _cmd = '3dNotes'
    input_spec = NotesInputSpec
    output_spec = AFNICommandOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.in_file)
        return outputs