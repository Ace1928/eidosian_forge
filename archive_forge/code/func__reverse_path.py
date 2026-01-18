from snappy.snap import t3mlite as t3m
from truncatedComplex import *
@staticmethod
def _reverse_path(path):
    return [edge.reverse() for edge in path[::-1]]