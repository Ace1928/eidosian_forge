import os
import sys
def _add_py_module(dist, ffi, module_name):
    from setuptools.command.build_py import build_py
    from setuptools.command.build_ext import build_ext
    from cffi._shimmed_dist_utils import log, mkpath
    from cffi import recompiler

    def generate_mod(py_file):
        log.info('generating cffi module %r' % py_file)
        mkpath(os.path.dirname(py_file))
        updated = recompiler.make_py_source(ffi, module_name, py_file)
        if not updated:
            log.info('already up-to-date')
    base_class = dist.cmdclass.get('build_py', build_py)

    class build_py_make_mod(base_class):

        def run(self):
            base_class.run(self)
            module_path = module_name.split('.')
            module_path[-1] += '.py'
            generate_mod(os.path.join(self.build_lib, *module_path))

        def get_source_files(self):
            saved_py_modules = self.py_modules
            try:
                if saved_py_modules:
                    self.py_modules = [m for m in saved_py_modules if m != module_name]
                return base_class.get_source_files(self)
            finally:
                self.py_modules = saved_py_modules
    dist.cmdclass['build_py'] = build_py_make_mod
    if dist.py_modules is None:
        dist.py_modules = []
    dist.py_modules.append(module_name)
    base_class_2 = dist.cmdclass.get('build_ext', build_ext)

    class build_ext_make_mod(base_class_2):

        def run(self):
            base_class_2.run(self)
            if self.inplace:
                module_path = module_name.split('.')
                package = '.'.join(module_path[:-1])
                build_py = self.get_finalized_command('build_py')
                package_dir = build_py.get_package_dir(package)
                file_name = module_path[-1] + '.py'
                generate_mod(os.path.join(package_dir, file_name))
    dist.cmdclass['build_ext'] = build_ext_make_mod