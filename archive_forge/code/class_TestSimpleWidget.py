import pytest
import os
import subprocess
import sys
import shutil
class TestSimpleWidget(PyinstallerBase):
    pinstall_path = os.path.join(os.path.dirname(__file__), 'simple_widget')