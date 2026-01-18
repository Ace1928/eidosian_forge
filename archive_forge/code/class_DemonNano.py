import os
import os.path as op
import pathlib as pl
import numpy as np
from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters
import ase.io
class DemonNano(FileIOCalculator):
    """Calculator interface to the deMon-nano code. """
    implemented_properties = ['energy', 'forces']

    def __init__(self, **kwargs):
        """ASE interface to the deMon-nano code.
        
        The deMon-nano code can be obtained from http://demon-nano.ups-tlse.fr/

        The ASE_DEMONNANO_COMMAND environment variable must be set to run the executable, in bash it would be set along the lines of
        export ASE_DEMONNANO_COMMAND="pathway-to-deMon-binary/deMon.username.x"

        Parameters:

        label : str 
            relative path to the run directory
        atoms  : Atoms object
            the atoms object
        command  : str
            Command to run deMon. If not present, the environment variable ASE_DEMONNANO_COMMAND is used
        basis_path  : str 
            Relative path to the directory containing DFTB-SCC or DFTB-0 parameters
            If not present, the environment variable DEMONNANO_BASIS_PATH is used
        restart_path  : str 
            Relative path to the deMon restart dir
        title : str 
            Title in the deMon input file.
        forces : bool
            If True a force calculation is enforced
        print_out : str | list 
            Options for the printing in deMon
        input_arguments : dict 
            Explicitly given input arguments. The key is the input keyword
            and the value is either a str, a list of str (will be written on the same line as the keyword),
            or a list of lists of str (first list is written on the first line, the others on following lines.)
        """
        parameters = DemonNanoParameters(**kwargs)
        basis_path = parameters['basis_path']
        if basis_path is None:
            basis_path = os.environ.get('DEMONNANO_BASIS_PATH')
        if basis_path is None:
            mess = 'The "DEMONNANO_BASIS_PATH" environment is not defined.'
            raise ValueError(mess)
        else:
            parameters['basis_path'] = basis_path
        FileIOCalculator.__init__(self, **parameters)

    def __getitem__(self, key):
        """Convenience method to retrieve a parameter as
        calculator[key] rather than calculator.parameters[key]

            Parameters:
                key       : str, the name of the parameters to get.
        """
        return self.parameters[key]

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input (in)-file.
        See calculator.py for further details.
 
        Parameters:
             atoms        : The Atoms object to write.
             properties   : The properties which should be calculated.
             system_changes : List of properties changed since last run.
        
        """
        FileIOCalculator.write_input(self, atoms=atoms, properties=properties, system_changes=system_changes)
        if system_changes is None and properties is None:
            return
        filename = self.label + '/deMon.inp'
        with open(filename, 'w') as fd:
            value = self.parameters['title']
            self._write_argument('TITLE', value, fd)
            fd.write('\n')
            value = self.parameters['forces']
            if 'forces' in properties or value:
                self._write_argument('MDYNAMICS', 'ZERO', fd)
                self._write_argument('MDSTEP', 'MAX=1', fd)
            value = self.parameters['print_out']
            assert isinstance(value, str)
            if not len(value) == 0:
                self._write_argument('PRINT', value, fd)
                fd.write('\n')
            self._write_input_arguments(fd)
            if 'BASISPATH' not in self.parameters['input_arguments']:
                value = self.parameters['basis_path']
                fd.write(value)
                fd.write('\n')
            self._write_atomic_coordinates(fd, atoms)
            ase.io.write(self.label + '/deMon_atoms.xyz', self.atoms)

    def read(self, restart_path):
        """Read parameters from directory restart_path."""
        self.set_label(restart_path)
        rpath = pl.Path(restart_path)
        if not (rpath / 'deMon.inp').exists():
            raise ReadError('The restart_path file {0} does not exist'.format(rpath))
        self.atoms = self.deMon_inp_to_atoms(rpath / 'deMon.inp')
        self.read_results()

    def _write_input_arguments(self, fd):
        """Write directly given input-arguments."""
        input_arguments = self.parameters['input_arguments']
        if input_arguments is None:
            return
        for key, value in input_arguments.items():
            self._write_argument(key, value, fd)

    def _write_argument(self, key, value, fd):
        """Write an argument to file.
       key :  a string coresponding to the input keyword
       value : the arguments, can be a string, a number or a list
       fd  :  and open file
       """
        if key == 'BASISPATH':
            line = value.lower()
            fd.write(line)
            fd.write('\n')
        elif not isinstance(value, (tuple, list)):
            line = key.upper()
            line += ' ' + str(value).upper()
            fd.write(line)
            fd.write('\n')
        else:
            line = key
            if not isinstance(value[0], (tuple, list)):
                for i in range(len(value)):
                    line += ' ' + str(value[i].upper())
                fd.write(line)
                fd.write('\n')
            else:
                for i in range(len(value)):
                    for j in range(len(value[i])):
                        line += ' ' + str(value[i][j]).upper()
                    fd.write(line)
                    fd.write('\n')
                    line = ''

    def _write_atomic_coordinates(self, fd, atoms):
        """Write atomic coordinates.
        Parameters:
        - fd:     An open file object.
        - atoms: An atoms object.
        """
        fd.write('GEOMETRY CARTESIAN ANGSTROM\n')
        for sym, pos in zip(atoms.symbols, atoms.positions):
            fd.write('{:9s} {:10.5f} {:10.5f} {:10.5f}\n'.format(sym, *pos))
        fd.write('\n')

    def read_results(self):
        """Read the results from output files."""
        self.read_energy()
        self.read_forces(self.atoms)

    def read_energy(self):
        """Read energy from deMon.ase output file."""
        epath = pl.Path(self.label)
        if not (epath / 'deMon.ase').exists():
            raise ReadError('The deMonNano output file for ASE {0} does not exist'.format(epath))
        filename = self.label + '/deMon.ase'
        if op.isfile(filename):
            with open(filename, 'r') as fd:
                lines = fd.readlines()
        for i in range(len(lines)):
            if lines[i].startswith(' DFTB total energy [Hartree]'):
                self.results['energy'] = float(lines[i + 1]) * Hartree
                break

    def read_forces(self, atoms):
        """Read forces from the deMon.ase file."""
        natoms = len(atoms)
        epath = pl.Path(self.label)
        if not (epath / 'deMon.ase').exists():
            raise ReadError('The deMonNano output file for ASE {0} does not exist'.format(epath))
        filename = self.label + '/deMon.ase'
        with open(filename, 'r') as fd:
            lines = fd.readlines()
            flag_found = False
            for i in range(len(lines)):
                if 'DFTB gradients at 0 time step in a.u.' in lines[i]:
                    start = i + 1
                    flag_found = True
                    break
            if flag_found:
                self.results['forces'] = np.zeros((natoms, 3), float)
                for i in range(natoms):
                    line = [s for s in lines[i + start].strip().split(' ') if len(s) > 0]
                    f = -np.array([float(x) for x in line[1:4]])
                    self.results['forces'][i, :] = f * (Hartree / Bohr)

    def deMon_inp_to_atoms(self, filename):
        """Routine to read deMon.inp and convert it to an atoms object."""
        read_flag = False
        chem_symbols = []
        xyz = []
        with open(filename, 'r') as fd:
            for line in fd:
                if 'GEOMETRY' in line:
                    read_flag = True
                    if 'ANGSTROM' in line:
                        coord_units = 'Ang'
                    elif 'BOHR' in line:
                        coord_units = 'Bohr'
                if read_flag:
                    tokens = line.split()
                    symbol = tokens[0]
                    xyz_loc = np.array(tokens[1:4]).astype(float)
                if read_flag and tokens:
                    chem_symbols.append(symbol)
                    xyz.append(xyz_loc)
        if coord_units == 'Bohr':
            xyz = xyz * Bohr
        atoms = ase.Atoms(symbols=chem_symbols, positions=xyz)
        return atoms