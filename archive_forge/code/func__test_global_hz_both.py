import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _test_global_hz_both(self, fname, type, ext, enum_test=True, dememorization=10000, batches=20, iterations=5000):
    """Use Global Hardy-Weinberg test for heterozygote deficiency/excess (PRIVATE).

        Returns a triple with:
         - A list per population containing (pop_name, P-val, SE, switches).
           Some pops have a None if the info is not available.
           SE might be none (for enumerations).
         - A list per loci containing (locus_name, P-val, SE, switches).
           Some loci have a None if the info is not available.
           SE might be none (for enumerations).
         - Overall results (P-val, SE, switches).

        """
    opts = self._get_opts(dememorization, batches, iterations, enum_test)
    self._run_genepop([ext], [1, type], fname, opts)

    def hw_pop_func(self):
        return _read_table(self.stream, [str, _gp_float, _gp_float, _gp_float])
    with open(fname + ext) as f1:
        line = f1.readline()
        while 'by population' not in line:
            line = f1.readline()
        pop_p = _read_table(f1, [str, _gp_float, _gp_float, _gp_float])
    with open(fname + ext) as f2:
        line = f2.readline()
        while 'by locus' not in line:
            line = f2.readline()
        loc_p = _read_table(f2, [str, _gp_float, _gp_float, _gp_float])
    with open(fname + ext) as f:
        line = f.readline()
        while 'all locus' not in line:
            line = f.readline()
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        line = f.readline().rstrip()
        p, se, switches = tuple((_gp_float(x) for x in [y for y in line.split(' ') if y != '']))
    return (pop_p, loc_p, (p, se, switches))