import pytest
import subprocess
import json
import sys
from numpy.distutils import _shell_utils
from numpy.testing import IS_WASM
@pytest.fixture(params=[_shell_utils.WindowsParser, _shell_utils.PosixParser])
def Parser(request):
    return request.param