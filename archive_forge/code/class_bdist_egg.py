from distutils.dir_util import remove_tree, mkpath
from distutils import log
from types import CodeType
import sys
import os
import re
import textwrap
import marshal
from setuptools.extension import Library
from setuptools import Command
from .._path import ensure_directory
from sysconfig import get_path, get_python_version
class bdist_egg(Command):
    description = 'create an "egg" distribution'
    user_options = [('bdist-dir=', 'b', 'temporary directory for creating the distribution'), ('plat-name=', 'p', 'platform name to embed in generated filenames (by default uses `pkg_resources.get_build_platform()`)'), ('exclude-source-files', None, 'remove all .py files from the generated egg'), ('keep-temp', 'k', 'keep the pseudo-installation tree around after ' + 'creating the distribution archive'), ('dist-dir=', 'd', 'directory to put final built distributions in'), ('skip-build', None, 'skip rebuilding everything (for testing/debugging)')]
    boolean_options = ['keep-temp', 'skip-build', 'exclude-source-files']

    def initialize_options(self):
        self.bdist_dir = None
        self.plat_name = None
        self.keep_temp = 0
        self.dist_dir = None
        self.skip_build = 0
        self.egg_output = None
        self.exclude_source_files = None

    def finalize_options(self):
        ei_cmd = self.ei_cmd = self.get_finalized_command('egg_info')
        self.egg_info = ei_cmd.egg_info
        if self.bdist_dir is None:
            bdist_base = self.get_finalized_command('bdist').bdist_base
            self.bdist_dir = os.path.join(bdist_base, 'egg')
        if self.plat_name is None:
            from pkg_resources import get_build_platform
            self.plat_name = get_build_platform()
        self.set_undefined_options('bdist', ('dist_dir', 'dist_dir'))
        if self.egg_output is None:
            basename = ei_cmd._get_egg_basename(py_version=get_python_version(), platform=self.distribution.has_ext_modules() and self.plat_name)
            self.egg_output = os.path.join(self.dist_dir, basename + '.egg')

    def do_install_data(self):
        self.get_finalized_command('install').install_lib = self.bdist_dir
        site_packages = os.path.normcase(os.path.realpath(_get_purelib()))
        old, self.distribution.data_files = (self.distribution.data_files, [])
        for item in old:
            if isinstance(item, tuple) and len(item) == 2:
                if os.path.isabs(item[0]):
                    realpath = os.path.realpath(item[0])
                    normalized = os.path.normcase(realpath)
                    if normalized == site_packages or normalized.startswith(site_packages + os.sep):
                        item = (realpath[len(site_packages) + 1:], item[1])
            self.distribution.data_files.append(item)
        try:
            log.info('installing package data to %s', self.bdist_dir)
            self.call_command('install_data', force=0, root=None)
        finally:
            self.distribution.data_files = old

    def get_outputs(self):
        return [self.egg_output]

    def call_command(self, cmdname, **kw):
        """Invoke reinitialized command `cmdname` with keyword args"""
        for dirname in INSTALL_DIRECTORY_ATTRS:
            kw.setdefault(dirname, self.bdist_dir)
        kw.setdefault('skip_build', self.skip_build)
        kw.setdefault('dry_run', self.dry_run)
        cmd = self.reinitialize_command(cmdname, **kw)
        self.run_command(cmdname)
        return cmd

    def run(self):
        self.run_command('egg_info')
        log.info('installing library code to %s', self.bdist_dir)
        instcmd = self.get_finalized_command('install')
        old_root = instcmd.root
        instcmd.root = None
        if self.distribution.has_c_libraries() and (not self.skip_build):
            self.run_command('build_clib')
        cmd = self.call_command('install_lib', warn_dir=0)
        instcmd.root = old_root
        all_outputs, ext_outputs = self.get_ext_outputs()
        self.stubs = []
        to_compile = []
        for p, ext_name in enumerate(ext_outputs):
            filename, ext = os.path.splitext(ext_name)
            pyfile = os.path.join(self.bdist_dir, strip_module(filename) + '.py')
            self.stubs.append(pyfile)
            log.info('creating stub loader for %s', ext_name)
            if not self.dry_run:
                write_stub(os.path.basename(ext_name), pyfile)
            to_compile.append(pyfile)
            ext_outputs[p] = ext_name.replace(os.sep, '/')
        if to_compile:
            cmd.byte_compile(to_compile)
        if self.distribution.data_files:
            self.do_install_data()
        archive_root = self.bdist_dir
        egg_info = os.path.join(archive_root, 'EGG-INFO')
        self.mkpath(egg_info)
        if self.distribution.scripts:
            script_dir = os.path.join(egg_info, 'scripts')
            log.info('installing scripts to %s', script_dir)
            self.call_command('install_scripts', install_dir=script_dir, no_ep=1)
        self.copy_metadata_to(egg_info)
        native_libs = os.path.join(egg_info, 'native_libs.txt')
        if all_outputs:
            log.info('writing %s', native_libs)
            if not self.dry_run:
                ensure_directory(native_libs)
                libs_file = open(native_libs, 'wt')
                libs_file.write('\n'.join(all_outputs))
                libs_file.write('\n')
                libs_file.close()
        elif os.path.isfile(native_libs):
            log.info('removing %s', native_libs)
            if not self.dry_run:
                os.unlink(native_libs)
        write_safety_flag(os.path.join(archive_root, 'EGG-INFO'), self.zip_safe())
        if os.path.exists(os.path.join(self.egg_info, 'depends.txt')):
            log.warn("WARNING: 'depends.txt' will not be used by setuptools 0.6!\nUse the install_requires/extras_require setup() args instead.")
        if self.exclude_source_files:
            self.zap_pyfiles()
        make_zipfile(self.egg_output, archive_root, verbose=self.verbose, dry_run=self.dry_run, mode=self.gen_header())
        if not self.keep_temp:
            remove_tree(self.bdist_dir, dry_run=self.dry_run)
        getattr(self.distribution, 'dist_files', []).append(('bdist_egg', get_python_version(), self.egg_output))

    def zap_pyfiles(self):
        log.info('Removing .py files from temporary directory')
        for base, dirs, files in walk_egg(self.bdist_dir):
            for name in files:
                path = os.path.join(base, name)
                if name.endswith('.py'):
                    log.debug('Deleting %s', path)
                    os.unlink(path)
                if base.endswith('__pycache__'):
                    path_old = path
                    pattern = '(?P<name>.+)\\.(?P<magic>[^.]+)\\.pyc'
                    m = re.match(pattern, name)
                    path_new = os.path.join(base, os.pardir, m.group('name') + '.pyc')
                    log.info('Renaming file from [%s] to [%s]' % (path_old, path_new))
                    try:
                        os.remove(path_new)
                    except OSError:
                        pass
                    os.rename(path_old, path_new)

    def zip_safe(self):
        safe = getattr(self.distribution, 'zip_safe', None)
        if safe is not None:
            return safe
        log.warn('zip_safe flag not set; analyzing archive contents...')
        return analyze_egg(self.bdist_dir, self.stubs)

    def gen_header(self):
        return 'w'

    def copy_metadata_to(self, target_dir):
        """Copy metadata (egg info) to the target_dir"""
        norm_egg_info = os.path.normpath(self.egg_info)
        prefix = os.path.join(norm_egg_info, '')
        for path in self.ei_cmd.filelist.files:
            if path.startswith(prefix):
                target = os.path.join(target_dir, path[len(prefix):])
                ensure_directory(target)
                self.copy_file(path, target)

    def get_ext_outputs(self):
        """Get a list of relative paths to C extensions in the output distro"""
        all_outputs = []
        ext_outputs = []
        paths = {self.bdist_dir: ''}
        for base, dirs, files in sorted_walk(self.bdist_dir):
            for filename in files:
                if os.path.splitext(filename)[1].lower() in NATIVE_EXTENSIONS:
                    all_outputs.append(paths[base] + filename)
            for filename in dirs:
                paths[os.path.join(base, filename)] = paths[base] + filename + '/'
        if self.distribution.has_ext_modules():
            build_cmd = self.get_finalized_command('build_ext')
            for ext in build_cmd.extensions:
                if isinstance(ext, Library):
                    continue
                fullname = build_cmd.get_ext_fullname(ext.name)
                filename = build_cmd.get_ext_filename(fullname)
                if not os.path.basename(filename).startswith('dl-'):
                    if os.path.exists(os.path.join(self.bdist_dir, filename)):
                        ext_outputs.append(filename)
        return (all_outputs, ext_outputs)