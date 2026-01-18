from __future__ import annotations
from typing import Callable, Optional
from collections import OrderedDict
import os
import re
import subprocess
from .util import (
class CompilerRunner:
    """ CompilerRunner base class.

    Parameters
    ==========

    sources : list of str
        Paths to sources.
    out : str
    flags : iterable of str
        Compiler flags.
    run_linker : bool
    compiler_name_exe : (str, str) tuple
        Tuple of compiler name &  command to call.
    cwd : str
        Path of root of relative paths.
    include_dirs : list of str
        Include directories.
    libraries : list of str
        Libraries to link against.
    library_dirs : list of str
        Paths to search for shared libraries.
    std : str
        Standard string, e.g. ``'c++11'``, ``'c99'``, ``'f2003'``.
    define: iterable of strings
        macros to define
    undef : iterable of strings
        macros to undefine
    preferred_vendor : string
        name of preferred vendor e.g. 'gnu' or 'intel'

    Methods
    =======

    run():
        Invoke compilation as a subprocess.

    """
    compiler_dict: dict[str, str]
    standards: tuple[None | str, ...]
    std_formater: dict[str, Callable[[Optional[str]], str]]
    compiler_name_vendor_mapping: dict[str, str]

    def __init__(self, sources, out, flags=None, run_linker=True, compiler=None, cwd='.', include_dirs=None, libraries=None, library_dirs=None, std=None, define=None, undef=None, strict_aliasing=None, preferred_vendor=None, linkline=None, **kwargs):
        if isinstance(sources, str):
            raise ValueError('Expected argument sources to be a list of strings.')
        self.sources = list(sources)
        self.out = out
        self.flags = flags or []
        self.cwd = cwd
        if compiler:
            self.compiler_name, self.compiler_binary = compiler
        else:
            if preferred_vendor is None:
                preferred_vendor = os.environ.get('SYMPY_COMPILER_VENDOR', None)
            self.compiler_name, self.compiler_binary, self.compiler_vendor = self.find_compiler(preferred_vendor)
            if self.compiler_binary is None:
                raise ValueError('No compiler found (searched: {})'.format(', '.join(self.compiler_dict.values())))
        self.define = define or []
        self.undef = undef or []
        self.include_dirs = include_dirs or []
        self.libraries = libraries or []
        self.library_dirs = library_dirs or []
        self.std = std or self.standards[0]
        self.run_linker = run_linker
        if self.run_linker:
            self.flags = list(filter(lambda x: x != '-c', self.flags))
        elif '-c' not in self.flags:
            self.flags.append('-c')
        if self.std:
            self.flags.append(self.std_formater[self.compiler_name](self.std))
        self.linkline = linkline or []
        if strict_aliasing is not None:
            nsa_re = re.compile('no-strict-aliasing$')
            sa_re = re.compile('strict-aliasing$')
            if strict_aliasing is True:
                if any(map(nsa_re.match, flags)):
                    raise CompileError('Strict aliasing cannot be both enforced and disabled')
                elif any(map(sa_re.match, flags)):
                    pass
                else:
                    flags.append('-fstrict-aliasing')
            elif strict_aliasing is False:
                if any(map(nsa_re.match, flags)):
                    pass
                elif any(map(sa_re.match, flags)):
                    raise CompileError('Strict aliasing cannot be both enforced and disabled')
                else:
                    flags.append('-fno-strict-aliasing')
            else:
                msg = 'Expected argument strict_aliasing to be True/False, got {}'
                raise ValueError(msg.format(strict_aliasing))

    @classmethod
    def find_compiler(cls, preferred_vendor=None):
        """ Identify a suitable C/fortran/other compiler. """
        candidates = list(cls.compiler_dict.keys())
        if preferred_vendor:
            if preferred_vendor in candidates:
                candidates = [preferred_vendor] + candidates
            else:
                raise ValueError('Unknown vendor {}'.format(preferred_vendor))
        name, path = find_binary_of_command([cls.compiler_dict[x] for x in candidates])
        return (name, path, cls.compiler_name_vendor_mapping[name])

    def cmd(self):
        """ List of arguments (str) to be passed to e.g. ``subprocess.Popen``. """
        cmd = [self.compiler_binary] + self.flags + ['-U' + x for x in self.undef] + ['-D' + x for x in self.define] + ['-I' + x for x in self.include_dirs] + self.sources
        if self.run_linker:
            cmd += ['-L' + x for x in self.library_dirs] + ['-l' + x for x in self.libraries] + self.linkline
        counted = []
        for envvar in re.findall('\\$\\{(\\w+)\\}', ' '.join(cmd)):
            if os.getenv(envvar) is None:
                if envvar not in counted:
                    counted.append(envvar)
                    msg = "Environment variable '{}' undefined.".format(envvar)
                    raise CompileError(msg)
        return cmd

    def run(self):
        self.flags = unique_list(self.flags)
        self.flags.extend(['-o', self.out])
        env = os.environ.copy()
        env['PWD'] = self.cwd
        p = subprocess.Popen(' '.join(self.cmd()), shell=True, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        comm = p.communicate()
        try:
            self.cmd_outerr = comm[0].decode('utf-8')
        except UnicodeDecodeError:
            self.cmd_outerr = comm[0].decode('iso-8859-1')
        self.cmd_returncode = p.returncode
        if self.cmd_returncode != 0:
            msg = "Error executing '{}' in {} (exited status {}):\n {}\n".format(' '.join(self.cmd()), self.cwd, str(self.cmd_returncode), self.cmd_outerr)
            raise CompileError(msg)
        return (self.cmd_outerr, self.cmd_returncode)