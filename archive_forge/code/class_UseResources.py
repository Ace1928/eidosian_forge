import os
import pytest
from .... import config
from ....utils.profiler import _use_resources
from ...base import traits, CommandLine, CommandLineInputSpec
from ... import utility as niu
class UseResources(CommandLine):
    """
    use_resources cmd interface
    """
    from nipype import __path__
    input_spec = UseResourcesInputSpec
    exec_dir = os.path.realpath(__path__[0])
    exec_path = os.path.join(exec_dir, 'utils', 'tests', 'use_resources')
    _cmd = exec_path
    _always_run = True