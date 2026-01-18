from __future__ import absolute_import, print_function
import os
import shutil
import tempfile
from .Dependencies import cythonize, extended_iglob
from ..Utils import is_package_dir
from ..Compiler import Options
def _cython_compile_files(all_paths, options):
    pool = None
    try:
        for path in all_paths:
            if options.build_inplace:
                base_dir = path
                while not os.path.isdir(base_dir) or is_package_dir(base_dir):
                    base_dir = os.path.dirname(base_dir)
            else:
                base_dir = None
            if os.path.isdir(path):
                paths = [os.path.join(path, '**', '*.{py,pyx}')]
            else:
                paths = [path]
            ext_modules = cythonize(paths, nthreads=options.parallel, exclude_failures=options.keep_going, exclude=options.excludes, compiler_directives=options.directives, compile_time_env=options.compile_time_env, force=options.force, quiet=options.quiet, depfile=options.depfile, language=options.language, **options.options)
            if ext_modules and options.build:
                if len(ext_modules) > 1 and options.parallel > 1:
                    if pool is None:
                        try:
                            pool = multiprocessing.Pool(options.parallel)
                        except OSError:
                            pool = _FakePool()
                    pool.map_async(run_distutils, [(base_dir, [ext]) for ext in ext_modules])
                else:
                    run_distutils((base_dir, ext_modules))
    except:
        if pool is not None:
            pool.terminate()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()