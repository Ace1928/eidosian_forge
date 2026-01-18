from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import os
import datetime
import traceback
import platform  # NOQA
from _ast import *  # NOQA
from ast import parse  # NOQA
from setuptools import setup, Extension, Distribution  # NOQA
from setuptools.command import install_lib  # NOQA
from setuptools.command.sdist import sdist as _sdist  # NOQA
@property
def _analyse_packages(self):
    """gather from configuration, names starting with * need
        to be installed explicitly as they are not on PyPI
        install_requires should be  dict, with keys 'any', 'py27' etc
        or a list (which is as if only 'any' was defined

        ToDo: update with: pep508 conditional dependencies
        """
    if self._pkg[0] is None:
        self._pkg[0] = []
        self._pkg[1] = []
    ir = self._pkg_data.get('install_requires')
    if ir is None:
        return self._pkg
    if isinstance(ir, list):
        self._pkg[0] = ir
        return self._pkg
    packages = ir.get('any', [])
    if isinstance(packages, string_type):
        packages = packages.split()
    if self.nested:
        parent_pkg = self.full_package_name.rsplit('.', 1)[0]
        if parent_pkg not in packages:
            packages.append(parent_pkg)
    implementation = platform.python_implementation()
    if implementation == 'CPython':
        pyver = 'py{0}{1}'.format(*sys.version_info)
    elif implementation == 'PyPy':
        pyver = 'pypy' if sys.version_info < (3,) else 'pypy3'
    elif implementation == 'Jython':
        pyver = 'jython'
    packages.extend(ir.get(pyver, []))
    for p in packages:
        if p[0] == '*':
            p = p[1:]
            self._pkg[1].append(p)
        self._pkg[0].append(p)
    return self._pkg