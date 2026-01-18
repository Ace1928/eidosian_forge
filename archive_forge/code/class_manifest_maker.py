from distutils.filelist import FileList as _FileList
from distutils.errors import DistutilsInternalError
from distutils.util import convert_path
from distutils import log
import distutils.errors
import distutils.filelist
import functools
import os
import re
import sys
import time
import collections
from .._importlib import metadata
from .. import _entry_points, _normalization
from . import _requirestxt
from setuptools import Command
from setuptools.command.sdist import sdist
from setuptools.command.sdist import walk_revctrl
from setuptools.command.setopt import edit_config
from setuptools.command import bdist_egg
import setuptools.unicode_utils as unicode_utils
from setuptools.glob import glob
from setuptools.extern import packaging
from ..warnings import SetuptoolsDeprecationWarning
class manifest_maker(sdist):
    template = 'MANIFEST.in'

    def initialize_options(self):
        self.use_defaults = 1
        self.prune = 1
        self.manifest_only = 1
        self.force_manifest = 1
        self.ignore_egg_info_dir = False

    def finalize_options(self):
        pass

    def run(self):
        self.filelist = FileList(ignore_egg_info_dir=self.ignore_egg_info_dir)
        if not os.path.exists(self.manifest):
            self.write_manifest()
        self.add_defaults()
        if os.path.exists(self.template):
            self.read_template()
        self.add_license_files()
        self._add_referenced_files()
        self.prune_file_list()
        self.filelist.sort()
        self.filelist.remove_duplicates()
        self.write_manifest()

    def _manifest_normalize(self, path):
        path = unicode_utils.filesys_decode(path)
        return path.replace(os.sep, '/')

    def write_manifest(self):
        """
        Write the file list in 'self.filelist' to the manifest file
        named by 'self.manifest'.
        """
        self.filelist._repair()
        files = [self._manifest_normalize(f) for f in self.filelist.files]
        msg = "writing manifest file '%s'" % self.manifest
        self.execute(write_file, (self.manifest, files), msg)

    def warn(self, msg):
        if not self._should_suppress_warning(msg):
            sdist.warn(self, msg)

    @staticmethod
    def _should_suppress_warning(msg):
        """
        suppress missing-file warnings from sdist
        """
        return re.match('standard file .*not found', msg)

    def add_defaults(self):
        sdist.add_defaults(self)
        self.filelist.append(self.template)
        self.filelist.append(self.manifest)
        rcfiles = list(walk_revctrl())
        if rcfiles:
            self.filelist.extend(rcfiles)
        elif os.path.exists(self.manifest):
            self.read_manifest()
        if os.path.exists('setup.py'):
            self.filelist.append('setup.py')
        ei_cmd = self.get_finalized_command('egg_info')
        self.filelist.graft(ei_cmd.egg_info)

    def add_license_files(self):
        license_files = self.distribution.metadata.license_files or []
        for lf in license_files:
            log.info("adding license file '%s'", lf)
        self.filelist.extend(license_files)

    def _add_referenced_files(self):
        """Add files referenced by the config (e.g. `file:` directive) to filelist"""
        referenced = getattr(self.distribution, '_referenced_files', [])
        for rf in referenced:
            log.debug("adding file referenced by config '%s'", rf)
        self.filelist.extend(referenced)

    def prune_file_list(self):
        build = self.get_finalized_command('build')
        base_dir = self.distribution.get_fullname()
        self.filelist.prune(build.build_base)
        self.filelist.prune(base_dir)
        sep = re.escape(os.sep)
        self.filelist.exclude_pattern('(^|' + sep + ')(RCS|CVS|\\.svn)' + sep, is_regex=1)

    def _safe_data_files(self, build_py):
        """
        The parent class implementation of this method
        (``sdist``) will try to include data files, which
        might cause recursion problems when
        ``include_package_data=True``.

        Therefore, avoid triggering any attempt of
        analyzing/building the manifest again.
        """
        if hasattr(build_py, 'get_data_files_without_manifest'):
            return build_py.get_data_files_without_manifest()
        SetuptoolsDeprecationWarning.emit("`build_py` command does not inherit from setuptools' `build_py`.", "\n            Custom 'build_py' does not implement 'get_data_files_without_manifest'.\n            Please extend command classes from setuptools instead of distutils.\n            ", see_url='https://peps.python.org/pep-0632/')
        return build_py.get_data_files()