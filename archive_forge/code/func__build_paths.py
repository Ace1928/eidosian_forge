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
def _build_paths(self, name, spec_path_lists, exists):
    """
        Given an environment variable name and specified paths,
        return a pathsep-separated string of paths containing
        unique, extant, directories from those paths and from
        the environment variable. Raise an error if no paths
        are resolved.

        Parameters
        ----------
        name: str
            Environment variable name
        spec_path_lists: list of str
            Paths
        exists: bool
            It True, only return existing paths.

        Return
        ------
        str
            Pathsep-separated paths
        """
    spec_paths = itertools.chain.from_iterable(spec_path_lists)
    env_paths = environ.get(name, '').split(pathsep)
    paths = itertools.chain(spec_paths, env_paths)
    extant_paths = list(filter(isdir, paths)) if exists else paths
    if not extant_paths:
        msg = '%s environment variable is empty' % name.upper()
        raise distutils.errors.DistutilsPlatformError(msg)
    unique_paths = unique_everseen(extant_paths)
    return pathsep.join(unique_paths)