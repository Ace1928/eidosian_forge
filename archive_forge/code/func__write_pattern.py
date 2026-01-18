from math import exp, pi, sin, sqrt, cos, acos
import numpy as np
from ase.data import atomic_numbers
def _write_pattern(self, fd):
    fd.write('# Wavelength = %f\n' % self.wavelength)
    if self.mode == 'XRD':
        x, y = (self.twotheta_list, self.intensity_list)
        fd.write('# 2theta \t Intesity\n')
    elif self.mode == 'SAXS':
        x, y = (self.q_list, self.intensity_list)
        fd.write('# q(1/A)\tIntesity\n')
    else:
        raise Exception('No data available, call calc_pattern() first.')
    for i in range(len(x)):
        fd.write('  %f\t%f\n' % (x[i], y[i]))