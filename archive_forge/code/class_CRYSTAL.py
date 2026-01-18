from ase.units import Hartree, Bohr
from ase.io import write
import numpy as np
import os
from ase.calculators.calculator import FileIOCalculator
class CRYSTAL(FileIOCalculator):
    """ A crystal calculator with ase-FileIOCalculator nomenclature
    """
    implemented_properties = ['energy', 'forces', 'stress', 'charges', 'dipole']

    def __init__(self, restart=None, ignore_bad_restart_file=FileIOCalculator._deprecated, label='cry', atoms=None, crys_pcc=False, **kwargs):
        """Construct a crystal calculator.

        """
        self.default_parameters = dict(xc='HF', spinpol=False, oldgrid=False, neigh=False, coarsegrid=False, guess=True, kpts=None, isp=1, basis='custom', smearing=None, otherkeys=[])
        self.pcpot = None
        self.lines = None
        self.atoms = None
        self.crys_pcc = crys_pcc
        self.atoms_input = None
        self.outfilename = 'cry.out'
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file, label, atoms, **kwargs)

    def write_crystal_in(self, filename):
        """ Write the input file for the crystal calculation.
            Geometry is taken always from the file 'fort.34'
        """
        with open(filename, 'wt', encoding='latin-1') as outfile:
            self._write_crystal_in(outfile)

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

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        self.write_crystal_in(os.path.join(self.directory, 'INPUT'))
        write(os.path.join(self.directory, 'fort.34'), atoms)
        self.atoms_input = atoms
        self.atoms = None

    def read_results(self):
        """ all results are read from OUTPUT file
            It will be destroyed after it is read to avoid
            reading it once again after some runtime error """
        with open(os.path.join(self.directory, 'OUTPUT'), 'rt', encoding='latin-1') as myfile:
            self.lines = myfile.readlines()
        self.atoms = self.atoms_input
        estring1 = 'SCF ENDED'
        estring2 = 'TOTAL ENERGY + DISP'
        for iline, line in enumerate(self.lines):
            if line.find(estring1) >= 0:
                index_energy = iline
                pos_en = 8
                break
        else:
            raise RuntimeError('Problem in reading energy')
        for iline, line in enumerate(self.lines):
            if line.find(estring2) >= 0:
                index_energy = iline
                pos_en = 5
        e_coul = 0
        if self.pcpot:
            if self.crys_pcc:
                self.pcpot.read_pc_corrections()
                self.pcpot.crys_pcc = True
            else:
                self.pcpot.manual_pc_correct()
            e_coul, f_coul = self.pcpot.coulomb_corrections
        energy = float(self.lines[index_energy].split()[pos_en]) * Hartree
        energy -= e_coul
        self.results['energy'] = energy
        fstring = 'CARTESIAN FORCES'
        gradients = []
        for iline, line in enumerate(self.lines):
            if line.find(fstring) >= 0:
                index_force_begin = iline + 2
                break
        else:
            raise RuntimeError('Problem in reading forces')
        for j in range(index_force_begin, index_force_begin + len(self.atoms)):
            word = self.lines[j].split()
            if len(word) == 5:
                gradients.append([float(word[k + 2]) for k in range(0, 3)])
            elif len(word) == 4:
                gradients.append([float(word[k + 1]) for k in range(0, 3)])
            else:
                raise RuntimeError('Problem in reading forces')
        forces = np.array(gradients) * Hartree / Bohr
        self.results['forces'] = forces
        sstring = 'STRESS TENSOR, IN'
        have_stress = False
        stress = []
        for iline, line in enumerate(self.lines):
            if sstring in line:
                have_stress = True
                start = iline + 4
                end = start + 3
                for i in range(start, end):
                    cell = [float(x) for x in self.lines[i].split()]
                    stress.append(cell)
        if have_stress:
            stress = -np.array(stress) * Hartree / Bohr ** 3
            self.results['stress'] = stress
        qm_charges = []
        for n, line in enumerate(self.lines):
            if 'TOTAL ATOMIC CHARGE' in line:
                chargestart = n + 1
        lines1 = self.lines[chargestart:chargestart + (len(self.atoms) - 1) // 6 + 1]
        atomnum = self.atoms.get_atomic_numbers()
        words = []
        for line in lines1:
            for el in line.split():
                words.append(float(el))
        i = 0
        for atn in atomnum:
            qm_charges.append(-words[i] + atn)
            i = i + 1
        charges = np.array(qm_charges)
        self.results['charges'] = charges
        dipole = np.zeros([1, 3])
        for n, line in enumerate(self.lines):
            if 'DIPOLE MOMENT ALONG' in line:
                dipolestart = n + 2
                dipole = np.array([float(f) for f in self.lines[dipolestart].split()[2:5]])
                break
        self.results['dipole'] = dipole * 0.2081943482534

    def embed(self, mmcharges=None, directory='./'):
        """Embed atoms in point-charges (mmcharges)
        """
        self.pcpot = PointChargePotential(mmcharges, self.directory)
        return self.pcpot