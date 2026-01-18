import os
from configparser import ConfigParser
from distutils import log
from distutils.command.build_py import build_py
from distutils.command.install_scripts import install_scripts
from distutils.version import LooseVersion
from os.path import join as pjoin
from os.path import split as psplit
from os.path import splitext
def _package_status(pkg_name, version, version_getter, checker):
    try:
        __import__(pkg_name)
    except ImportError:
        return ('missing', None)
    if not version:
        return ('satisfied', None)
    try:
        have_version = version_getter(pkg_name)
    except AttributeError:
        return ('no-version', None)
    if checker(have_version) < checker(version):
        return ('low-version', have_version)
    return ('satisfied', have_version)