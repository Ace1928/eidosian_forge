import os
import pytest
from .... import config
from ....utils.profiler import _use_resources
from ...base import traits, CommandLine, CommandLineInputSpec
from ... import utility as niu
class UseResourcesInputSpec(CommandLineInputSpec):
    mem_gb = traits.Float(desc='Number of GB of RAM to use', argstr='-g %f', mandatory=True)
    n_procs = traits.Int(desc='Number of threads to use', argstr='-p %d', mandatory=True)