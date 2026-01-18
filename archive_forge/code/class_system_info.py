import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
class system_info:
    """ get_info() is the only public method. Don't use others.
    """
    dir_env_var = None
    search_static_first = 0
    section = 'ALL'
    saved_results = {}
    notfounderror = NotFoundError

    def __init__(self, default_lib_dirs=default_lib_dirs, default_include_dirs=default_include_dirs):
        self.__class__.info = {}
        self.local_prefixes = []
        defaults = {'library_dirs': os.pathsep.join(default_lib_dirs), 'include_dirs': os.pathsep.join(default_include_dirs), 'runtime_library_dirs': os.pathsep.join(default_runtime_dirs), 'rpath': '', 'src_dirs': os.pathsep.join(default_src_dirs), 'search_static_first': str(self.search_static_first), 'extra_compile_args': '', 'extra_link_args': ''}
        self.cp = ConfigParser(defaults)
        self.files = []
        self.files.extend(get_standard_file('.numpy-site.cfg'))
        self.files.extend(get_standard_file('site.cfg'))
        self.parse_config_files()
        if self.section is not None:
            self.search_static_first = self.cp.getboolean(self.section, 'search_static_first')
        assert isinstance(self.search_static_first, int)

    def parse_config_files(self):
        self.cp.read(self.files)
        if not self.cp.has_section(self.section):
            if self.section is not None:
                self.cp.add_section(self.section)

    def calc_libraries_info(self):
        libs = self.get_libraries()
        dirs = self.get_lib_dirs()
        r_dirs = self.get_runtime_lib_dirs()
        r_dirs.extend(self.get_runtime_lib_dirs(key='rpath'))
        info = {}
        for lib in libs:
            i = self.check_libs(dirs, [lib])
            if i is not None:
                dict_append(info, **i)
            else:
                log.info('Library %s was not found. Ignoring' % lib)
            if r_dirs:
                i = self.check_libs(r_dirs, [lib])
                if i is not None:
                    del i['libraries']
                    i['runtime_library_dirs'] = i.pop('library_dirs')
                    dict_append(info, **i)
                else:
                    log.info('Runtime library %s was not found. Ignoring' % lib)
        return info

    def set_info(self, **info):
        if info:
            lib_info = self.calc_libraries_info()
            dict_append(info, **lib_info)
            extra_info = self.calc_extra_info()
            dict_append(info, **extra_info)
        self.saved_results[self.__class__.__name__] = info

    def get_option_single(self, *options):
        """ Ensure that only one of `options` are found in the section

        Parameters
        ----------
        *options : list of str
           a list of options to be found in the section (``self.section``)

        Returns
        -------
        str :
            the option that is uniquely found in the section

        Raises
        ------
        AliasedOptionError :
            in case more than one of the options are found
        """
        found = [self.cp.has_option(self.section, opt) for opt in options]
        if sum(found) == 1:
            return options[found.index(True)]
        elif sum(found) == 0:
            return options[0]
        if AliasedOptionError.__doc__ is None:
            raise AliasedOptionError()
        raise AliasedOptionError(AliasedOptionError.__doc__.format(section=self.section, options='[{}]'.format(', '.join(options))))

    def has_info(self):
        return self.__class__.__name__ in self.saved_results

    def calc_extra_info(self):
        """ Updates the information in the current information with
        respect to these flags:
          extra_compile_args
          extra_link_args
        """
        info = {}
        for key in ['extra_compile_args', 'extra_link_args']:
            opt = self.cp.get(self.section, key)
            opt = _shell_utils.NativeParser.split(opt)
            if opt:
                tmp = {key: opt}
                dict_append(info, **tmp)
        return info

    def get_info(self, notfound_action=0):
        """ Return a dictionary with items that are compatible
            with numpy.distutils.setup keyword arguments.
        """
        flag = 0
        if not self.has_info():
            flag = 1
            log.info(self.__class__.__name__ + ':')
            if hasattr(self, 'calc_info'):
                self.calc_info()
            if notfound_action:
                if not self.has_info():
                    if notfound_action == 1:
                        warnings.warn(self.notfounderror.__doc__, stacklevel=2)
                    elif notfound_action == 2:
                        raise self.notfounderror(self.notfounderror.__doc__)
                    else:
                        raise ValueError(repr(notfound_action))
            if not self.has_info():
                log.info('  NOT AVAILABLE')
                self.set_info()
            else:
                log.info('  FOUND:')
        res = self.saved_results.get(self.__class__.__name__)
        if log.get_threshold() <= log.INFO and flag:
            for k, v in res.items():
                v = str(v)
                if k in ['sources', 'libraries'] and len(v) > 270:
                    v = v[:120] + '...\n...\n...' + v[-120:]
                log.info('    %s = %s', k, v)
            log.info('')
        return copy.deepcopy(res)

    def get_paths(self, section, key):
        dirs = self.cp.get(section, key).split(os.pathsep)
        env_var = self.dir_env_var
        if env_var:
            if is_sequence(env_var):
                e0 = env_var[-1]
                for e in env_var:
                    if e in os.environ:
                        e0 = e
                        break
                if not env_var[0] == e0:
                    log.info('Setting %s=%s' % (env_var[0], e0))
                env_var = e0
        if env_var and env_var in os.environ:
            d = os.environ[env_var]
            if d == 'None':
                log.info('Disabled %s: %s', self.__class__.__name__, '(%s is None)' % (env_var,))
                return []
            if os.path.isfile(d):
                dirs = [os.path.dirname(d)] + dirs
                l = getattr(self, '_lib_names', [])
                if len(l) == 1:
                    b = os.path.basename(d)
                    b = os.path.splitext(b)[0]
                    if b[:3] == 'lib':
                        log.info('Replacing _lib_names[0]==%r with %r' % (self._lib_names[0], b[3:]))
                        self._lib_names[0] = b[3:]
            else:
                ds = d.split(os.pathsep)
                ds2 = []
                for d in ds:
                    if os.path.isdir(d):
                        ds2.append(d)
                        for dd in ['include', 'lib']:
                            d1 = os.path.join(d, dd)
                            if os.path.isdir(d1):
                                ds2.append(d1)
                dirs = ds2 + dirs
        default_dirs = self.cp.get(self.section, key).split(os.pathsep)
        dirs.extend(default_dirs)
        ret = []
        for d in dirs:
            if len(d) > 0 and (not os.path.isdir(d)):
                warnings.warn('Specified path %s is invalid.' % d, stacklevel=2)
                continue
            if d not in ret:
                ret.append(d)
        log.debug('( %s = %s )', key, ':'.join(ret))
        return ret

    def get_lib_dirs(self, key='library_dirs'):
        return self.get_paths(self.section, key)

    def get_runtime_lib_dirs(self, key='runtime_library_dirs'):
        path = self.get_paths(self.section, key)
        if path == ['']:
            path = []
        return path

    def get_include_dirs(self, key='include_dirs'):
        return self.get_paths(self.section, key)

    def get_src_dirs(self, key='src_dirs'):
        return self.get_paths(self.section, key)

    def get_libs(self, key, default):
        try:
            libs = self.cp.get(self.section, key)
        except NoOptionError:
            if not default:
                return []
            if is_string(default):
                return [default]
            return default
        return [b for b in [a.strip() for a in libs.split(',')] if b]

    def get_libraries(self, key='libraries'):
        if hasattr(self, '_lib_names'):
            return self.get_libs(key, default=self._lib_names)
        else:
            return self.get_libs(key, '')

    def library_extensions(self):
        c = customized_ccompiler()
        static_exts = []
        if c.compiler_type != 'msvc':
            static_exts.append('.a')
        if sys.platform == 'win32':
            static_exts.append('.lib')
        if self.search_static_first:
            exts = static_exts + [so_ext]
        else:
            exts = [so_ext] + static_exts
        if sys.platform == 'cygwin':
            exts.append('.dll.a')
        if sys.platform == 'darwin':
            exts.append('.dylib')
        return exts

    def check_libs(self, lib_dirs, libs, opt_libs=[]):
        """If static or shared libraries are available then return
        their info dictionary.

        Checks for all libraries as shared libraries first, then
        static (or vice versa if self.search_static_first is True).
        """
        exts = self.library_extensions()
        info = None
        for ext in exts:
            info = self._check_libs(lib_dirs, libs, opt_libs, [ext])
            if info is not None:
                break
        if not info:
            log.info('  libraries %s not found in %s', ','.join(libs), lib_dirs)
        return info

    def check_libs2(self, lib_dirs, libs, opt_libs=[]):
        """If static or shared libraries are available then return
        their info dictionary.

        Checks each library for shared or static.
        """
        exts = self.library_extensions()
        info = self._check_libs(lib_dirs, libs, opt_libs, exts)
        if not info:
            log.info('  libraries %s not found in %s', ','.join(libs), lib_dirs)
        return info

    def _find_lib(self, lib_dir, lib, exts):
        assert is_string(lib_dir)
        if sys.platform == 'win32':
            lib_prefixes = ['', 'lib']
        else:
            lib_prefixes = ['lib']
        for ext in exts:
            for prefix in lib_prefixes:
                p = self.combine_paths(lib_dir, prefix + lib + ext)
                if p:
                    break
            if p:
                assert len(p) == 1
                if ext == '.dll.a':
                    lib += '.dll'
                if ext == '.lib':
                    lib = prefix + lib
                return lib
        return False

    def _find_libs(self, lib_dirs, libs, exts):
        found_dirs, found_libs = ([], [])
        for lib in libs:
            for lib_dir in lib_dirs:
                found_lib = self._find_lib(lib_dir, lib, exts)
                if found_lib:
                    found_libs.append(found_lib)
                    if lib_dir not in found_dirs:
                        found_dirs.append(lib_dir)
                    break
        return (found_dirs, found_libs)

    def _check_libs(self, lib_dirs, libs, opt_libs, exts):
        """Find mandatory and optional libs in expected paths.

        Missing optional libraries are silently forgotten.
        """
        if not is_sequence(lib_dirs):
            lib_dirs = [lib_dirs]
        found_dirs, found_libs = self._find_libs(lib_dirs, libs, exts)
        if len(found_libs) > 0 and len(found_libs) == len(libs):
            opt_found_dirs, opt_found_libs = self._find_libs(lib_dirs, opt_libs, exts)
            found_libs.extend(opt_found_libs)
            for lib_dir in opt_found_dirs:
                if lib_dir not in found_dirs:
                    found_dirs.append(lib_dir)
            info = {'libraries': found_libs, 'library_dirs': found_dirs}
            return info
        else:
            return None

    def combine_paths(self, *args):
        """Return a list of existing paths composed by all combinations
        of items from the arguments.
        """
        return combine_paths(*args)