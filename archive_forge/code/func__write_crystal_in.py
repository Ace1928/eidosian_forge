from ase.units import Hartree, Bohr
from ase.io import write
import numpy as np
import os
from ase.calculators.calculator import FileIOCalculator
def _write_crystal_in(self, outfile):
    outfile.write('Single point + Gradient crystal calculation \n')
    outfile.write('EXTERNAL \n')
    outfile.write('NEIGHPRT \n')
    outfile.write('0 \n')
    if self.pcpot:
        outfile.write('POINTCHG \n')
        self.pcpot.write_mmcharges('POINTCHG.INP')
    p = self.parameters
    if p.basis == 'custom':
        outfile.write('END \n')
        with open(os.path.join(self.directory, 'basis')) as basisfile:
            basis_ = basisfile.readlines()
        for line in basis_:
            outfile.write(line)
        outfile.write('99 0 \n')
        outfile.write('END \n')
    else:
        outfile.write('BASISSET \n')
        outfile.write(p.basis.upper() + '\n')
    if self.atoms.get_initial_magnetic_moments().any():
        p.spinpol = True
    if p.xc == 'HF':
        if p.spinpol:
            outfile.write('UHF \n')
        else:
            outfile.write('RHF \n')
    elif p.xc == 'MP2':
        outfile.write('MP2 \n')
        outfile.write('ENDMP2 \n')
    else:
        outfile.write('DFT \n')
        if isinstance(p.xc, str):
            xc = {'LDA': 'EXCHANGE\nLDA\nCORRELAT\nVWN', 'PBE': 'PBEXC'}.get(p.xc, p.xc)
            outfile.write(xc.upper() + '\n')
        else:
            x, c = p.xc
            outfile.write('EXCHANGE \n')
            outfile.write(x + ' \n')
            outfile.write('CORRELAT \n')
            outfile.write(c + ' \n')
        if p.spinpol:
            outfile.write('SPIN \n')
        if p.oldgrid:
            outfile.write('OLDGRID \n')
        if p.coarsegrid:
            outfile.write('RADIAL\n')
            outfile.write('1\n')
            outfile.write('4.0\n')
            outfile.write('20\n')
            outfile.write('ANGULAR\n')
            outfile.write('5\n')
            outfile.write('0.1667 0.5 0.9 3.05 9999.0\n')
            outfile.write('2 6 8 13 8\n')
        outfile.write('END \n')
    if p.guess:
        if os.path.isfile('fort.20'):
            outfile.write('GUESSP \n')
        elif os.path.isfile('fort.9'):
            outfile.write('GUESSP \n')
            os.system('cp fort.9 fort.20')
    if p.smearing is not None:
        if p.smearing[0] != 'Fermi-Dirac':
            raise ValueError('Only Fermi-Dirac smearing is allowed.')
        else:
            outfile.write('SMEAR \n')
            outfile.write(str(p.smearing[1] / Hartree) + ' \n')
    for keyword in p.otherkeys:
        if isinstance(keyword, str):
            outfile.write(keyword.upper() + '\n')
        else:
            for key in keyword:
                outfile.write(key.upper() + '\n')
    ispbc = self.atoms.get_pbc()
    self.kpts = p.kpts
    if any(ispbc):
        if self.kpts is None:
            self.kpts = (1, 1, 1)
    else:
        self.kpts = None
    if self.kpts is not None:
        if isinstance(self.kpts, float):
            raise ValueError('K-point density definition not allowed.')
        if isinstance(self.kpts, list):
            raise ValueError('Explicit K-points definition not allowed.')
        if isinstance(self.kpts[-1], str):
            raise ValueError('Shifted Monkhorst-Pack not allowed.')
        outfile.write('SHRINK  \n')
        outfile.write('0 ' + str(p.isp * max(self.kpts)) + ' \n')
        if ispbc[2]:
            outfile.write(str(self.kpts[0]) + ' ' + str(self.kpts[1]) + ' ' + str(self.kpts[2]) + ' \n')
        elif ispbc[1]:
            outfile.write(str(self.kpts[0]) + ' ' + str(self.kpts[1]) + ' 1 \n')
        elif ispbc[0]:
            outfile.write(str(self.kpts[0]) + ' 1 1 \n')
    outfile.write('GRADCAL \n')
    outfile.write('END \n')