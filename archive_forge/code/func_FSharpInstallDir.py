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
@property
def FSharpInstallDir(self):
    """
        Microsoft Visual F# directory.

        Return
        ------
        str
            path
        """
    path = join(self.ri.visualstudio, '%0.1f\\Setup\\F#' % self.vs_ver)
    return self.ri.lookup(path, 'productdir') or ''