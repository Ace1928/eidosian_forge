import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def _setup_pip(self, context):
    """Installs or upgrades pip in a virtual environment"""
    self._call_new_python(context, '-m', 'ensurepip', '--upgrade', '--default-pip', stderr=subprocess.STDOUT)