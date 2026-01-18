import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _calc_diversities_fis(self, fname, ext):
    self._run_genepop([ext], [5, 2], fname)
    with open(fname + ext) as f:
        line = f.readline()
        while line != '':
            line = line.rstrip()
            if line.startswith('Statistics per sample over all loci with at least two individuals typed'):
                avg_fis = _read_table(f, [str, _gp_float, _gp_float, _gp_float])
                avg_Qintra = _read_table(f, [str, _gp_float])
            line = f.readline()

    def fis_func(self):
        line = self.stream.readline()
        while line != '':
            line = line.rstrip()
            m = re.search('Locus: (.+)', line)
            if m is not None:
                locus = m.group(1)
                self.stream.readline()
                if 'No complete' in self.stream.readline():
                    return (locus, None)
                self.stream.readline()
                fis_table = _read_table(self.stream, [str, _gp_float, _gp_float, _gp_float])
                self.stream.readline()
                avg_qinter, avg_fis = tuple((_gp_float(x) for x in [y for y in self.stream.readline().split(' ') if y != '']))
                return (locus, fis_table, avg_qinter, avg_fis)
            line = self.stream.readline()
        self.done = True
        raise StopIteration
    return (_FileIterator(fis_func, fname + ext), avg_fis, avg_Qintra)