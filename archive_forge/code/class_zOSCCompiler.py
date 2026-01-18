import os
from . import sysconfig
from .errors import CompileError, DistutilsExecError
from .unixccompiler import UnixCCompiler
class zOSCCompiler(UnixCCompiler):
    src_extensions = ['.c', '.C', '.cc', '.cxx', '.cpp', '.m', '.s']
    _cpp_extensions = ['.cc', '.cpp', '.cxx', '.C']
    _asm_extensions = ['.s']

    def _get_zos_compiler_name(self):
        zos_compiler_names = [os.path.basename(binary) for envvar in ('CC', 'CXX', 'LDSHARED') if (binary := os.environ.get(envvar, None))]
        if len(zos_compiler_names) == 0:
            return 'ibm-openxl'
        zos_compilers = {}
        for compiler in ('ibm-clang', 'ibm-clang64', 'ibm-clang++', 'ibm-clang++64', 'clang', 'clang++', 'clang-14'):
            zos_compilers[compiler] = 'ibm-openxl'
        for compiler in ('xlclang', 'xlclang++', 'njsc', 'njsc++'):
            zos_compilers[compiler] = 'ibm-xlclang'
        for compiler in ('xlc', 'xlC', 'xlc++'):
            zos_compilers[compiler] = 'ibm-xlc'
        return zos_compilers.get(zos_compiler_names[0], 'ibm-openxl')

    def __init__(self, verbose=0, dry_run=0, force=0):
        super().__init__(verbose, dry_run, force)
        self.zos_compiler = self._get_zos_compiler_name()
        sysconfig.customize_compiler(self)

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        local_args = []
        if ext in self._cpp_extensions:
            compiler = self.compiler_cxx
            local_args.extend(_cxx_args[self.zos_compiler])
        elif ext in self._asm_extensions:
            compiler = self.compiler_so
            local_args.extend(_cc_args[self.zos_compiler])
            local_args.extend(_asm_args[self.zos_compiler])
        else:
            compiler = self.compiler_so
            local_args.extend(_cc_args[self.zos_compiler])
        local_args.extend(cc_args)
        try:
            self.spawn(compiler + local_args + [src, '-o', obj] + extra_postargs)
        except DistutilsExecError as msg:
            raise CompileError(msg)

    def runtime_library_dir_option(self, dir):
        return '-L' + dir

    def link(self, target_desc, objects, output_filename, output_dir=None, libraries=None, library_dirs=None, runtime_library_dirs=None, export_symbols=None, debug=0, extra_preargs=None, extra_postargs=None, build_temp=None, target_lang=None):
        ldversion = sysconfig.get_config_var('LDVERSION')
        if sysconfig.python_build:
            side_deck_path = os.path.join(sysconfig.get_config_var('abs_builddir'), f'libpython{ldversion}.x')
        else:
            side_deck_path = os.path.join(sysconfig.get_config_var('installed_base'), sysconfig.get_config_var('platlibdir'), f'libpython{ldversion}.x')
        if os.path.exists(side_deck_path):
            if extra_postargs:
                extra_postargs.append(side_deck_path)
            else:
                extra_postargs = [side_deck_path]
        if runtime_library_dirs:
            for dir in runtime_library_dirs:
                for library in libraries[:]:
                    library_side_deck = os.path.join(dir, f'{library}.x')
                    if os.path.exists(library_side_deck):
                        libraries.remove(library)
                        extra_postargs.append(library_side_deck)
                        break
        extra_postargs.extend(_ld_args[self.zos_compiler])
        super().link(target_desc, objects, output_filename, output_dir, libraries, library_dirs, runtime_library_dirs, export_symbols, debug, extra_preargs, extra_postargs, build_temp, target_lang)