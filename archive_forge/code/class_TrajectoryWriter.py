import warnings
from typing import Tuple
import numpy as np
from ase import __version__
from ase.calculators.singlepoint import SinglePointCalculator, all_properties
from ase.constraints import dict2constraint
from ase.calculators.calculator import PropertyNotImplementedError
from ase.atoms import Atoms
from ase.io.jsonio import encode, decode
from ase.io.pickletrajectory import PickleTrajectory
from ase.parallel import world
from ase.utils import tokenize_version
class TrajectoryWriter:
    """Writes Atoms objects to a .traj file."""

    def __init__(self, filename, mode='w', atoms=None, properties=None, extra=[], master=None):
        """A Trajectory writer, in write or append mode.

        Parameters:

        filename: str
            The name of the file.  Traditionally ends in .traj.
        mode: str
            The mode.  'r' is read mode, the file should already exist, and
            no atoms argument should be specified.
            'w' is write mode.  The atoms argument specifies the Atoms
            object to be written to the file, if not given it must instead
            be given as an argument to the write() method.
            'a' is append mode.  It acts as write mode, except that
            data is appended to a preexisting file.
        atoms: Atoms object
            The Atoms object to be written in write or append mode.
        properties: list of str
            If specified, these calculator properties are saved in the
            trajectory.  If not specified, all supported quantities are
            saved.  Possible values: energy, forces, stress, dipole,
            charges, magmom and magmoms.
        master: bool
            Controls which process does the actual writing. The
            default is that process number 0 does this.  If this
            argument is given, processes where it is True will write.
        """
        if master is None:
            master = world.rank == 0
        self.master = master
        self.atoms = atoms
        self.properties = properties
        self.description = {}
        self.header_data = None
        self.multiple_headers = False
        self._open(filename, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def set_description(self, description):
        self.description.update(description)

    def _open(self, filename, mode):
        import ase.io.ulm as ulm
        if mode not in 'aw':
            raise ValueError('mode must be "w" or "a".')
        if self.master:
            self.backend = ulm.open(filename, mode, tag='ASE-Trajectory')
            if len(self.backend) > 0 and mode == 'a':
                with Trajectory(filename) as traj:
                    atoms = traj[0]
                self.header_data = get_header_data(atoms)
        else:
            self.backend = ulm.DummyWriter()

    def write(self, atoms=None, **kwargs):
        """Write the atoms to the file.

        If the atoms argument is not given, the atoms object specified
        when creating the trajectory object is used.

        Use keyword arguments to add extra properties::

            writer.write(atoms, energy=117, dipole=[0, 0, 1.0])
        """
        if atoms is None:
            atoms = self.atoms
        for image in atoms.iterimages():
            self._write_atoms(image, **kwargs)

    def _write_atoms(self, atoms, **kwargs):
        b = self.backend
        if self.header_data is None:
            b.write(version=1, ase_version=__version__)
            if self.description:
                b.write(description=self.description)
            self.header_data = get_header_data(atoms)
            write_header = True
        else:
            if not self.multiple_headers:
                header_data = get_header_data(atoms)
                self.multiple_headers = not headers_equal(self.header_data, header_data)
            write_header = self.multiple_headers
        write_atoms(b, atoms, write_header=write_header)
        calc = atoms.calc
        if calc is None and len(kwargs) > 0:
            calc = SinglePointCalculator(atoms)
        if calc is not None:
            if not hasattr(calc, 'get_property'):
                calc = OldCalculatorWrapper(calc)
            c = b.child('calculator')
            c.write(name=calc.name)
            if hasattr(calc, 'todict'):
                c.write(parameters=calc.todict())
            for prop in all_properties:
                if prop in kwargs:
                    x = kwargs[prop]
                elif self.properties is not None:
                    if prop in self.properties:
                        x = calc.get_property(prop, atoms)
                    else:
                        x = None
                else:
                    try:
                        x = calc.get_property(prop, atoms, allow_calculation=False)
                    except (PropertyNotImplementedError, KeyError):
                        x = None
                if x is not None:
                    if prop in ['stress', 'dipole']:
                        x = x.tolist()
                    c.write(prop, x)
        info = {}
        for key, value in atoms.info.items():
            try:
                encode(value)
            except TypeError:
                warnings.warn('Skipping "{0}" info.'.format(key))
            else:
                info[key] = value
        if info:
            b.write(info=info)
        b.sync()

    def close(self):
        """Close the trajectory file."""
        self.backend.close()

    def __len__(self):
        return world.sum(len(self.backend))