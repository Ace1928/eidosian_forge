import builtins
import os
import sys
import platform
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch
from IPython.core import debugger
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
from IPython.testing.decorators import skip_win32
import pytest
def _decorator_skip_setup():
    import pexpect
    env = os.environ.copy()
    env['IPY_TEST_SIMPLE_PROMPT'] = '1'
    env['PROMPT_TOOLKIT_NO_CPR'] = '1'
    child = pexpect.spawn(sys.executable, ['-m', 'IPython', '--colors=nocolor'], env=env)
    child.timeout = 15 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.expect('IPython')
    child.expect('\n')
    child.timeout = 5 * IPYTHON_TESTING_TIMEOUT_SCALE
    child.str_last_chars = 500
    dedented_blocks = [dedent(b).strip() for b in skip_decorators_blocks]
    in_prompt_number = 1
    for cblock in dedented_blocks:
        child.expect_exact(f'In [{in_prompt_number}]:')
        in_prompt_number += 1
        for line in cblock.splitlines():
            child.sendline(line)
            child.expect_exact(line)
        child.sendline('')
    return child