import base64
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pytest
from jupyter_console.ptshell import ZMQTerminalInteractiveShell
class NonCommunicatingShell(ZMQTerminalInteractiveShell):
    """A testing shell class that doesn't attempt to communicate with the kernel"""

    def init_kernel_info(self):
        pass