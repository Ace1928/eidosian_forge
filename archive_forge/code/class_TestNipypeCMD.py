import pytest
import sys
from contextlib import contextmanager
from io import StringIO
from ...utils import nipype_cmd
class TestNipypeCMD:
    maxDiff = None

    def test_main_returns_2_on_empty(self):
        with pytest.raises(SystemExit) as cm:
            with capture_sys_output() as (stdout, stderr):
                nipype_cmd.main(['nipype_cmd'])
        exit_exception = cm.value
        assert exit_exception.code == 2
        msg = 'usage: nipype_cmd [-h] module interface\nnipype_cmd: error: the following arguments are required: module, interface\n'
        assert stderr.getvalue() == msg
        assert stdout.getvalue() == ''

    def test_main_returns_0_on_help(self):
        with pytest.raises(SystemExit) as cm:
            with capture_sys_output() as (stdout, stderr):
                nipype_cmd.main(['nipype_cmd', '-h'])
        exit_exception = cm.value
        assert exit_exception.code == 0
        assert stderr.getvalue() == ''
        if sys.version_info >= (3, 10):
            options = 'options'
        else:
            options = 'optional arguments'
        assert stdout.getvalue() == f'usage: nipype_cmd [-h] module interface\n\nNipype interface runner\n\npositional arguments:\n  module      Module name\n  interface   Interface name\n\n{options}:\n  -h, --help  show this help message and exit\n'

    def test_list_nipy_interfacesp(self):
        with pytest.raises(SystemExit) as cm:
            with capture_sys_output() as (stdout, stderr):
                nipype_cmd.main(['nipype_cmd', 'nipype.interfaces.nipy'])
        with pytest.raises(SystemExit) as cm:
            with capture_sys_output() as (stdout, stderr):
                nipype_cmd.main(['nipype_cmd', 'nipype.interfaces.nipy'])
        exit_exception = cm.value
        assert exit_exception.code == 0
        assert stderr.getvalue() == ''
        assert stdout.getvalue() == 'Available Interfaces:\n\tComputeMask\n\tEstimateContrast\n\tFitGLM\n\tSimilarity\n\tSpaceTimeRealigner\n'