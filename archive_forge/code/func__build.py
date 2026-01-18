import contextlib
import functools
import io
import os
import shutil
import subprocess
import sys
import sysconfig
import setuptools
def _build(name, src, srcdir):
    if is_hip():
        hip_lib_dir = os.path.join(rocm_path_dir(), 'lib')
        hip_include_dir = os.path.join(rocm_path_dir(), 'include')
    else:
        cuda_lib_dirs = libcuda_dirs()
        cu_include_dir = cuda_include_dir()
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    cc = os.environ.get('CC')
    if cc is None:
        clang = shutil.which('clang')
        gcc = shutil.which('gcc')
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError('Failed to find C compiler. Please specify via CC environment variable.')
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)['include']
    if is_hip():
        ret = subprocess.check_call([cc, src, f'-I{hip_include_dir}', f'-I{py_include_dir}', f'-I{srcdir}', '-shared', '-fPIC', f'-L{hip_lib_dir}', '-lamdhip64', '-o', so])
    else:
        cc_cmd = [cc, src, '-O3', f'-I{cu_include_dir}', f'-I{py_include_dir}', f'-I{srcdir}', '-shared', '-fPIC', '-lcuda', '-o', so]
        cc_cmd += [f'-L{dir}' for dir in cuda_lib_dirs]
        ret = subprocess.check_call(cc_cmd)
    if ret == 0:
        return so
    extra_compile_args = []
    library_dirs = cuda_lib_dirs
    include_dirs = [srcdir, cu_include_dir]
    libraries = ['cuda']
    extra_link_args = []
    ext = setuptools.Extension(name=name, language='c', sources=[src], include_dirs=include_dirs, extra_compile_args=extra_compile_args + ['-O3'], extra_link_args=extra_link_args, library_dirs=library_dirs, libraries=libraries)
    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(name=name, ext_modules=[ext], script_args=args)
    with quiet():
        setuptools.setup(**args)
    return so