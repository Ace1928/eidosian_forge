import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion
class Gnu95FCompiler(GnuFCompiler):
    compiler_type = 'gnu95'
    compiler_aliases = ('gfortran',)
    description = 'GNU Fortran 95 compiler'

    def version_match(self, version_string):
        v = self.gnu_version_match(version_string)
        if not v or v[0] != 'gfortran':
            return None
        v = v[1]
        if LooseVersion(v) >= '4':
            pass
        elif sys.platform == 'win32':
            for key in ['version_cmd', 'compiler_f77', 'compiler_f90', 'compiler_fix', 'linker_so', 'linker_exe']:
                self.executables[key].append('-mno-cygwin')
        return v
    possible_executables = ['gfortran', 'f95']
    executables = {'version_cmd': ['<F90>', '-dumpversion'], 'compiler_f77': [None, '-Wall', '-g', '-ffixed-form', '-fno-second-underscore'], 'compiler_f90': [None, '-Wall', '-g', '-fno-second-underscore'], 'compiler_fix': [None, '-Wall', '-g', '-ffixed-form', '-fno-second-underscore'], 'linker_so': ['<F90>', '-Wall', '-g'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib'], 'linker_exe': [None, '-Wall']}
    module_dir_switch = '-J'
    module_include_switch = '-I'
    if sys.platform.startswith(('aix', 'os400')):
        executables['linker_so'].append('-lpthread')
        if platform.architecture()[0][:2] == '64':
            for key in ['compiler_f77', 'compiler_f90', 'compiler_fix', 'linker_so', 'linker_exe']:
                executables[key].append('-maix64')
    g2c = 'gfortran'

    def _universal_flags(self, cmd):
        """Return a list of -arch flags for every supported architecture."""
        if not sys.platform == 'darwin':
            return []
        arch_flags = []
        c_archs = self._c_arch_flags()
        if 'i386' in c_archs:
            c_archs[c_archs.index('i386')] = 'i686'
        for arch in ['ppc', 'i686', 'x86_64', 'ppc64', 's390x']:
            if _can_target(cmd, arch) and arch in c_archs:
                arch_flags.extend(['-arch', arch])
        return arch_flags

    def get_flags(self):
        flags = GnuFCompiler.get_flags(self)
        arch_flags = self._universal_flags(self.compiler_f90)
        if arch_flags:
            flags[:0] = arch_flags
        return flags

    def get_flags_linker_so(self):
        flags = GnuFCompiler.get_flags_linker_so(self)
        arch_flags = self._universal_flags(self.linker_so)
        if arch_flags:
            flags[:0] = arch_flags
        return flags

    def get_library_dirs(self):
        opt = GnuFCompiler.get_library_dirs(self)
        if sys.platform == 'win32':
            c_compiler = self.c_compiler
            if c_compiler and c_compiler.compiler_type == 'msvc':
                target = self.get_target()
                if target:
                    d = os.path.normpath(self.get_libgcc_dir())
                    root = os.path.join(d, *(os.pardir,) * 4)
                    path = os.path.join(root, 'lib')
                    mingwdir = os.path.normpath(path)
                    if os.path.exists(os.path.join(mingwdir, 'libmingwex.a')):
                        opt.append(mingwdir)
        lib_gfortran_dir = self.get_libgfortran_dir()
        if lib_gfortran_dir:
            opt.append(lib_gfortran_dir)
        return opt

    def get_libraries(self):
        opt = GnuFCompiler.get_libraries(self)
        if sys.platform == 'darwin':
            opt.remove('cc_dynamic')
        if sys.platform == 'win32':
            c_compiler = self.c_compiler
            if c_compiler and c_compiler.compiler_type == 'msvc':
                if 'gcc' in opt:
                    i = opt.index('gcc')
                    opt.insert(i + 1, 'mingwex')
                    opt.insert(i + 1, 'mingw32')
            c_compiler = self.c_compiler
            if c_compiler and c_compiler.compiler_type == 'msvc':
                return []
            else:
                pass
        return opt

    def get_target(self):
        try:
            p = subprocess.Popen(self.compiler_f77 + ['-v'], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate()
            output = (stdout or b'') + (stderr or b'')
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            output = filepath_from_subprocess_output(output)
            m = TARGET_R.search(output)
            if m:
                return m.group(1)
        return ''

    def _hash_files(self, filenames):
        h = hashlib.sha1()
        for fn in filenames:
            with open(fn, 'rb') as f:
                while True:
                    block = f.read(131072)
                    if not block:
                        break
                    h.update(block)
        text = base64.b32encode(h.digest())
        text = text.decode('ascii')
        return text.rstrip('=')

    def _link_wrapper_lib(self, objects, output_dir, extra_dll_dir, chained_dlls, is_archive):
        """Create a wrapper shared library for the given objects

        Return an MSVC-compatible lib
        """
        c_compiler = self.c_compiler
        if c_compiler.compiler_type != 'msvc':
            raise ValueError('This method only supports MSVC')
        object_hash = self._hash_files(list(objects) + list(chained_dlls))
        if is_win64():
            tag = 'win_amd64'
        else:
            tag = 'win32'
        basename = 'lib' + os.path.splitext(os.path.basename(objects[0]))[0][:8]
        root_name = basename + '.' + object_hash + '.gfortran-' + tag
        dll_name = root_name + '.dll'
        def_name = root_name + '.def'
        lib_name = root_name + '.lib'
        dll_path = os.path.join(extra_dll_dir, dll_name)
        def_path = os.path.join(output_dir, def_name)
        lib_path = os.path.join(output_dir, lib_name)
        if os.path.isfile(lib_path):
            return (lib_path, dll_path)
        if is_archive:
            objects = ['-Wl,--whole-archive'] + list(objects) + ['-Wl,--no-whole-archive']
        self.link_shared_object(objects, dll_name, output_dir=extra_dll_dir, extra_postargs=list(chained_dlls) + ['-Wl,--allow-multiple-definition', '-Wl,--output-def,' + def_path, '-Wl,--export-all-symbols', '-Wl,--enable-auto-import', '-static', '-mlong-double-64'])
        if is_win64():
            specifier = '/MACHINE:X64'
        else:
            specifier = '/MACHINE:X86'
        lib_args = ['/def:' + def_path, '/OUT:' + lib_path, specifier]
        if not c_compiler.initialized:
            c_compiler.initialize()
        c_compiler.spawn([c_compiler.lib] + lib_args)
        return (lib_path, dll_path)

    def can_ccompiler_link(self, compiler):
        return compiler.compiler_type not in ('msvc',)

    def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
        """
        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.
        """
        if self.c_compiler.compiler_type == 'msvc':
            archives = []
            plain_objects = []
            for obj in objects:
                if obj.lower().endswith('.a'):
                    archives.append(obj)
                else:
                    plain_objects.append(obj)
            chained_libs = []
            chained_dlls = []
            for archive in archives[::-1]:
                lib, dll = self._link_wrapper_lib([archive], output_dir, extra_dll_dir, chained_dlls=chained_dlls, is_archive=True)
                chained_libs.insert(0, lib)
                chained_dlls.insert(0, dll)
            if not plain_objects:
                return chained_libs
            lib, dll = self._link_wrapper_lib(plain_objects, output_dir, extra_dll_dir, chained_dlls=chained_dlls, is_archive=False)
            return [lib] + chained_libs
        else:
            raise ValueError('Unsupported C compiler')