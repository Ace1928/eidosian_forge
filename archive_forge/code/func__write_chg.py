import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
def _write_chg(self, fobj, chg, volume, format='chg'):
    """Write charge density

        Utility function similar to _read_chg but for writing.

        """
    chgtmp = chg.T.ravel()
    chgtmp = chgtmp * volume
    chgtmp = tuple(chgtmp)
    if format.lower() == 'chg':
        for ii in range((len(chgtmp) - 1) // 10):
            fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G\n' % chgtmp[ii * 10:(ii + 1) * 10])
        if len(chgtmp) % 10 == 0:
            fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G' % chgtmp[len(chgtmp) - 10:len(chgtmp)])
        else:
            for ii in range(len(chgtmp) % 10):
                fobj.write(' %#11.5G' % chgtmp[len(chgtmp) - len(chgtmp) % 10 + ii])
    else:
        for ii in range((len(chgtmp) - 1) // 5):
            fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E\n' % chgtmp[ii * 5:(ii + 1) * 5])
        if len(chgtmp) % 5 == 0:
            fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E' % chgtmp[len(chgtmp) - 5:len(chgtmp)])
        else:
            for ii in range(len(chgtmp) % 5):
                fobj.write(' %17.10E' % chgtmp[len(chgtmp) - len(chgtmp) % 5 + ii])
    fobj.write('\n')