import numpy as np
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import SCFError
import ase.units
class QChem(FileIOCalculator):
    """
    QChem calculator
    """
    name = 'QChem'
    implemented_properties = ['energy', 'forces']
    command = 'qchem PREFIX.inp PREFIX.out'
    default_parameters = {'method': 'hf', 'basis': '6-31G*', 'jobtype': None, 'charge': 0}

    def __init__(self, restart=None, ignore_bad_restart_file=FileIOCalculator._deprecated, label='qchem', scratch=None, np=1, nt=1, pbs=False, basisfile=None, ecpfile=None, atoms=None, **kwargs):
        """
        The scratch directory, number of processor and threads as well as a few
        other command line options can be set using the arguments explained
        below. The remaining kwargs are copied as options to the input file.
        The calculator will convert these options to upper case
        (Q-Chem standard) when writing the input file.

        scratch: str
            path of the scratch directory
        np: int
            number of processors for the -np command line flag
        nt: int
            number of threads for the -nt command line flag
        pbs: boolean
            command line flag for pbs scheduler (see Q-Chem manual)
        basisfile: str
            path to file containing the basis. Use in combination with
            basis='gen' keyword argument.
        ecpfile: str
            path to file containing the effective core potential. Use in
            combination with ecp='gen' keyword argument.
        """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file, label, atoms, **kwargs)
        if pbs:
            self.command = 'qchem -pbs '
        else:
            self.command = 'qchem '
        if np != 1:
            self.command += '-np %d ' % np
        if nt != 1:
            self.command += '-nt %d ' % nt
        self.command += 'PREFIX.inp PREFIX.out'
        if scratch is not None:
            self.command += ' %s' % scratch
        self.basisfile = basisfile
        self.ecpfile = ecpfile

    def read(self, label):
        raise NotImplementedError

    def read_results(self):
        filename = self.label + '.out'
        with open(filename, 'r') as fileobj:
            lineiter = iter(fileobj)
            for line in lineiter:
                if 'SCF failed to converge' in line:
                    raise SCFError()
                elif 'ERROR: alpha_min' in line:
                    raise SCFError()
                elif ' Total energy in the final basis set =' in line:
                    convert = ase.units.Hartree
                    self.results['energy'] = float(line.split()[8]) * convert
                elif ' Gradient of SCF Energy' in line:
                    gradient = [[] for _ in range(3)]
                    next(lineiter)
                    while True:
                        for i in range(3):
                            line = next(lineiter)[5:].rstrip()
                            gradient[i].extend(list(map(float, [line[i:i + 12] for i in range(0, len(line), 12)])))
                        if ' Max gradient component' in next(lineiter):
                            self.results['forces'] = np.array(gradient).T * (-ase.units.Hartree / ase.units.Bohr)
                            break

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        filename = self.label + '.inp'
        with open(filename, 'w') as fileobj:
            fileobj.write('$comment\n   ASE generated input file\n$end\n\n')
            fileobj.write('$rem\n')
            if self.parameters['jobtype'] is None:
                if 'forces' in properties:
                    fileobj.write('   %-25s   %s\n' % ('JOBTYPE', 'FORCE'))
                else:
                    fileobj.write('   %-25s   %s\n' % ('JOBTYPE', 'SP'))
            for prm in self.parameters:
                if prm not in ['charge', 'multiplicity']:
                    if self.parameters[prm] is not None:
                        fileobj.write('   %-25s   %s\n' % (prm.upper(), self.parameters[prm].upper()))
            fileobj.write('   %-25s   %s\n' % ('SYM_IGNORE', 'TRUE'))
            fileobj.write('$end\n\n')
            fileobj.write('$molecule\n')
            if 'multiplicity' not in self.parameters:
                tot_magmom = atoms.get_initial_magnetic_moments().sum()
                mult = tot_magmom + 1
            else:
                mult = self.parameters['multiplicity']
            fileobj.write('   %d %d\n' % (self.parameters['charge'], mult))
            for a in atoms:
                fileobj.write('   %s  %f  %f  %f\n' % (a.symbol, a.x, a.y, a.z))
            fileobj.write('$end\n\n')
            if self.basisfile is not None:
                with open(self.basisfile, 'r') as f_in:
                    basis = f_in.readlines()
                fileobj.write('$basis\n')
                fileobj.writelines(basis)
                fileobj.write('$end\n\n')
            if self.ecpfile is not None:
                with open(self.ecpfile, 'r') as f_in:
                    ecp = f_in.readlines()
                fileobj.write('$ecp\n')
                fileobj.writelines(ecp)
                fileobj.write('$end\n\n')