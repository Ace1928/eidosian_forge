import subprocess
from collections import namedtuple
def _naacl2pair(pair_string):
    i, j, p = pair_string.split('-')
    return (int(i), int(j))