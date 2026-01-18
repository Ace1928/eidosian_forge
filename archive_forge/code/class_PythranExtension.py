import pythran.config as cfg
from collections import defaultdict
import os.path
import os
class PythranExtension(Extension):
    """
    Description of a Pythran extension

    Similar to distutils.core.Extension except that the sources are .py files
    They must be processable by pythran, of course.

    The compilation process ends up in a native Python module.
    """

    def __init__(self, name, sources, *args, **kwargs):
        cfg_ext = cfg.make_extension(python=True, **kwargs)
        self.cxx = cfg_ext.pop('cxx', None)
        self.cc = cfg_ext.pop('cc', None)
        self._sources = sources
        Extension.__init__(self, name, sources, *args, **cfg_ext)
        self.__dict__.pop('sources', None)

    @property
    def sources(self):
        import pythran.toolchain as tc
        cxx_sources = []
        for source in self._sources:
            base, ext = os.path.splitext(source)
            if ext != '.py':
                cxx_sources.append(source)
                continue
            output_file = base + '.cpp'
            if os.path.exists(source) and (not os.path.exists(output_file) or os.path.getmtime(output_file) < os.path.getmtime(source)):
                if '.' in self.name:
                    module_name = os.path.splitext(self.name)[-1][1:]
                else:
                    module_name = self.name
                tc.compile_pythranfile(source, output_file, module_name, cpponly=True)
            cxx_sources.append(output_file)
        return cxx_sources

    @sources.setter
    def sources(self, sources):
        self._sources = sources