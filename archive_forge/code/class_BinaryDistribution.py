import fnmatch
import os
import platform
import re
import sys
from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution
class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True