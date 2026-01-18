import os
import re
import sys
import ctypes
import ctypes.util
import pyglet
class LinuxLibraryLoader(LibraryLoader):
    _ld_so_cache = None
    _local_libs_cache = None

    @staticmethod
    def _find_libs(directories):
        cache = {}
        lib_re = re.compile('lib(.*)\\.so(?:$|\\.)')
        for directory in directories:
            try:
                for file in os.listdir(directory):
                    match = lib_re.match(file)
                    if match:
                        path = os.path.join(directory, file)
                        if file not in cache:
                            cache[file] = path
                        library = match.group(1)
                        if library not in cache:
                            cache[library] = path
            except OSError:
                pass
        return cache

    def _create_ld_so_cache(self):
        directories = []
        try:
            directories.extend(os.environ['LD_LIBRARY_PATH'].split(':'))
        except KeyError:
            pass
        try:
            with open('/etc/ld.so.conf') as fid:
                directories.extend([directory.strip() for directory in fid])
        except IOError:
            pass
        directories.extend(['/lib', '/usr/lib'])
        self._ld_so_cache = self._find_libs(directories)

    def find_library(self, path):
        if _local_lib_paths:
            if not self._local_libs_cache:
                self._local_libs_cache = self._find_libs(_local_lib_paths)
            if path in self._local_libs_cache:
                return self._local_libs_cache[path]
        result = ctypes.util.find_library(path)
        if result:
            return result
        if self._ld_so_cache is None:
            self._create_ld_so_cache()
        return self._ld_so_cache.get(path)