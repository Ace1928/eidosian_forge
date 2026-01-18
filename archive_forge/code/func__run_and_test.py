import os
from copy import deepcopy
import pytest
import pdb
from nipype.utils.filemanip import split_filename, ensure_list
from .. import preprocess as fsl
from nipype.interfaces.fsl import Info
from nipype.interfaces.base import File, TraitError, Undefined, isdefined
from nipype.interfaces.fsl import no_fsl
def _run_and_test(opts, output_base):
    outputs = fsl.FAST(**opts)._list_outputs()
    for output in outputs.values():
        if output:
            for filename in ensure_list(output):
                assert os.path.realpath(filename).startswith(os.path.realpath(output_base))