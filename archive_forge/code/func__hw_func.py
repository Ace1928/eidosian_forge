import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _hw_func(stream, is_locus, has_fisher=False):
    line = stream.readline()
    if is_locus:
        hook = 'Locus '
    else:
        hook = 'Pop : '
    while line != '':
        if line.lstrip().startswith(hook):
            stream.readline()
            stream.readline()
            stream.readline()
            table = _read_table(stream, [str, _gp_float, _gp_float, _gp_float, _gp_float, _gp_int, str])
            loci = {}
            for entry in table:
                if len(entry) < 4:
                    loci[entry[0]] = None
                else:
                    locus, p, se, fis_wc, fis_rh, steps = entry[:-1]
                    if se == '-':
                        se = None
                    loci[locus] = (p, se, fis_wc, fis_rh, steps)
            return loci
        line = stream.readline()
    raise StopIteration