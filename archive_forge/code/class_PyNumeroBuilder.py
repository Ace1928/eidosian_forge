import sys
from pyomo.common.cmake_builder import build_cmake_project
class PyNumeroBuilder(object):

    def __call__(self, parallel):
        return build_pynumero(parallel=parallel)