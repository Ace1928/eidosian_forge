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
class InstallHeaders(Command):
    """Override how headers are copied.

  The install_headers that comes with setuptools copies all files to
  the same directory. But we need the files to be in a specific directory
  hierarchy for -I <include_dir> to work correctly.
  """
    description = 'install C/C++ header files'
    user_options = [('install-dir=', 'd', 'directory to install header files to'), ('force', 'f', 'force installation (overwrite existing files)')]
    boolean_options = ['force']

    def initialize_options(self):
        self.install_dir = None
        self.force = 0
        self.outfiles = []

    def finalize_options(self):
        self.set_undefined_options('install', ('install_headers', 'install_dir'), ('force', 'force'))

    def mkdir_and_copy_file(self, header):
        install_dir = os.path.join(self.install_dir, os.path.dirname(header))
        install_dir = re.sub('/google/protobuf_archive/src', '', install_dir)
        external_header_locations = ['tensorflow/include/external/eigen_archive/', 'tensorflow/include/external/com_google_absl/']
        for location in external_header_locations:
            if location in install_dir:
                extra_dir = install_dir.replace(location, '')
                if not os.path.exists(extra_dir):
                    self.mkpath(extra_dir)
                self.copy_file(header, extra_dir)
        if not os.path.exists(install_dir):
            self.mkpath(install_dir)
        return self.copy_file(header, install_dir)

    def run(self):
        hdrs = self.distribution.headers
        if not hdrs:
            return
        self.mkpath(self.install_dir)
        for header in hdrs:
            out, _ = self.mkdir_and_copy_file(header)
            self.outfiles.append(out)

    def get_inputs(self):
        return self.distribution.headers or []

    def get_outputs(self):
        return self.outfiles