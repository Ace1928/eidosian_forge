import os
import importlib.util
import sys
import glob
from distutils.core import Command
from distutils.errors import *
from distutils.util import convert_path, Mixin2to3
from distutils import log
class build_py_2to3(build_py, Mixin2to3):

    def run(self):
        self.updated_files = []
        if self.py_modules:
            self.build_modules()
        if self.packages:
            self.build_packages()
            self.build_package_data()
        self.run_2to3(self.updated_files)
        self.byte_compile(self.get_outputs(include_bytecode=0))

    def build_module(self, module, module_file, package):
        res = build_py.build_module(self, module, module_file, package)
        if res[1]:
            self.updated_files.append(res[0])
        return res