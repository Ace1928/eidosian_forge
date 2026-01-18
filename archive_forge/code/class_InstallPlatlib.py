import os
import sys
import json
from os.path import isdir
import subprocess
import pathlib
import setuptools
from setuptools import Command
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from distutils.command.build import build
from setuptools.dist import Distribution
import shutil
class InstallPlatlib(install):

    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib