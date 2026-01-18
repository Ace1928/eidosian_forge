import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from os.path import join as pjoin
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import pytest
from jupyter_core import paths
from jupyterlab import commands
from jupyterlab.commands import (
from jupyterlab.coreconfig import CoreConfig, _get_default_core_data
def _mock_extract(self, source, tempdir, *args, **kwargs):
    data = {'name': source, 'version': '2.1.0', 'jupyterlab': {'extension': True}, 'jupyterlab_extracted_files': ['index.js']}
    data.update(_gen_dep('^2000.0.0'))
    info = {'source': source, 'is_dir': False, 'data': data, 'name': source, 'version': data['version'], 'filename': 'mockextension.tgz', 'path': pjoin(tempdir, 'mockextension.tgz')}
    return info