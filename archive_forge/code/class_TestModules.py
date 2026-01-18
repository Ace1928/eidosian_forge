import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
class TestModules(tt.TempFileMixin):

    def test_extraneous_loads(self):
        """Test we're not loading modules on startup that we shouldn't.
        """
        self.mktmp("import sys\nprint('numpy' in sys.modules)\nprint('ipyparallel' in sys.modules)\nprint('ipykernel' in sys.modules)\n")
        out = 'False\nFalse\nFalse\n'
        tt.ipexec_validate(self.fname, out)