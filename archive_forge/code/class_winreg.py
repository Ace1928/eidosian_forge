import json
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
from subprocess import CalledProcessError
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.more_itertools import unique_everseen
class winreg:
    HKEY_USERS = None
    HKEY_CURRENT_USER = None
    HKEY_LOCAL_MACHINE = None
    HKEY_CLASSES_ROOT = None