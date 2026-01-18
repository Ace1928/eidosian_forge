from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import subprocess
import importlib
import pkg_resources
import threading
from subprocess import check_output
from dataclasses import dataclass
from typing import Optional
from fileio import File, PathIO, PathIOLike
from lazyops.envs import logger
from lazyops.envs import LazyEnv
class LazyImporter:
    libs = {}
    submodules = {}

    def __init__(self):
        self.lock = threading.RLock()

    @property
    def imports(self):
        return LazyImporter.libs

    def has_submodule(name, *args, **kwargs):
        bool(LazyImporter.get_submodule(name))

    def get_submodule(name, *args, **kwargs):
        return LazyImporter.submodules.get(name, None)

    def setup_submodule(self, name, *args, **kwargs):
        self.lock.acquire()
        with self.lock:
            if name not in LazyImporter.submodules:
                LazyImporter.submodules[name] = LazySubmodule(*args, name=name, **kwargs)
                LazyImporter.submodules[name].lazy_init()
            return LazyImporter.submodules[name]

    def setup_lib(self, name, lib_name=None, force=False, latest=False, verbose=False):
        self.lock.acquire()
        with self.lock:
            if name not in LazyImporter.libs:
                LazyImporter.libs[name] = lazy_init(name, lib_name, force, latest, verbose)
            return LazyImporter.libs[name]

    def has_lib(self, name, lib_name=None):
        return bool(self.get_lib(name, lib_name))

    def get_lib(self, name, lib_name=None):
        return LazyImporter.libs.get(name, LazyImporter.libs.get(lib_name, None))

    def __call__(self, name, lib_name=None, *args, **kwargs):
        if self.has_lib(name, lib_name):
            return self.get_lib(name, lib_name)
        return self.setup_lib(name, *args, lib_name=lib_name, **kwargs)