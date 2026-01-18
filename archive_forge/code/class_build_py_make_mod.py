import os
import sys
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