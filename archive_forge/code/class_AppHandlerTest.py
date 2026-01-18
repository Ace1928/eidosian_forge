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
class AppHandlerTest(TestCase):

    def tempdir(self):
        td = TemporaryDirectory()
        self.tempdirs.append(td)
        return td.name

    def setUp(self):
        self.tempdirs = []
        self.devnull = open(os.devnull, 'w')

        @self.addCleanup
        def cleanup_tempdirs():
            for d in self.tempdirs:
                d.cleanup()
        self.test_dir = self.tempdir()
        self.data_dir = pjoin(self.test_dir, 'data')
        self.config_dir = pjoin(self.test_dir, 'config')
        self.pkg_names = {}
        for name in ['extension', 'incompat', 'package', 'mimeextension']:
            src = pjoin(here, 'mock_packages', name)

            def ignore(dname, files):
                if 'node_modules' in dname:
                    files = []
                if 'node_modules' in files:
                    files.remove('node_modules')
                return (dname, files)
            dest = pjoin(self.test_dir, name)
            shutil.copytree(src, dest, ignore=ignore)
            if not os.path.exists(pjoin(dest, 'node_modules')):
                os.makedirs(pjoin(dest, 'node_modules'))
            setattr(self, 'mock_' + name, dest)
            with open(pjoin(dest, 'package.json')) as fid:
                data = json.load(fid)
            self.pkg_names[name] = data['name']
        self.patches = []
        p = patch.dict('os.environ', {'JUPYTER_CONFIG_DIR': self.config_dir, 'JUPYTER_DATA_DIR': self.data_dir, 'JUPYTERLAB_DIR': pjoin(self.data_dir, 'lab')})
        self.patches.append(p)
        for mod in [paths]:
            if hasattr(mod, 'ENV_JUPYTER_PATH'):
                p = patch.object(mod, 'ENV_JUPYTER_PATH', [self.data_dir])
                self.patches.append(p)
            if hasattr(mod, 'ENV_CONFIG_PATH'):
                p = patch.object(mod, 'ENV_CONFIG_PATH', [self.config_dir])
                self.patches.append(p)
            if hasattr(mod, 'CONFIG_PATH'):
                p = patch.object(mod, 'CONFIG_PATH', self.config_dir)
                self.patches.append(p)
            if hasattr(mod, 'BUILD_PATH'):
                p = patch.object(mod, 'BUILD_PATH', self.data_dir)
                self.patches.append(p)
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.assertEqual(paths.ENV_CONFIG_PATH, [self.config_dir])
        self.assertEqual(paths.ENV_JUPYTER_PATH, [self.data_dir])
        self.assertEqual(Path(commands.get_app_dir()).resolve(), (Path(self.data_dir) / 'lab').resolve())
        self.app_dir = commands.get_app_dir()
        self.pinned_packages = ['jupyterlab-test-extension@1.0', 'jupyterlab-test-extension@2.0']