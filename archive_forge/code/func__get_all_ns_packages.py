import os
from distutils import log
import itertools
def _get_all_ns_packages(self):
    """Return sorted list of all package namespaces"""
    pkgs = self.distribution.namespace_packages or []
    return sorted(set(flatten(map(self._pkg_names, pkgs))))