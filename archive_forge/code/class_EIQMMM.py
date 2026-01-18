import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
class EIQMMM(Calculator, IOContext):
    """Explicit interaction QMMM calculator."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, selection, qmcalc, mmcalc, interaction, vacuum=None, embedding=None, output=None):
        """EIQMMM object.

        The energy is calculated as::

                    _          _         _    _
            E = E  (R  ) + E  (R  ) + E (R  , R  )
                 QM  QM     MM  MM     I  QM   MM

        parameters:

        selection: list of int, slice object or list of bool
            Selection out of all the atoms that belong to the QM part.
        qmcalc: Calculator object
            QM-calculator.
        mmcalc: Calculator object
            MM-calculator.
        interaction: Interaction object
            Interaction between QM and MM regions.
        vacuum: float or None
            Amount of vacuum to add around QM atoms.  Use None if QM
            calculator doesn't need a box.
        embedding: Embedding object or None
            Specialized embedding object.  Use None in order to use the
            default one.
        output: None, '-', str or file-descriptor.
            File for logging information - default is no logging (None).

        """
        self.selection = selection
        self.qmcalc = qmcalc
        self.mmcalc = mmcalc
        self.interaction = interaction
        self.vacuum = vacuum
        self.embedding = embedding
        self.qmatoms = None
        self.mmatoms = None
        self.mask = None
        self.center = None
        self.name = '{0}+{1}+{2}'.format(qmcalc.name, interaction.name, mmcalc.name)
        self.output = self.openfile(output)
        Calculator.__init__(self)

    def initialize(self, atoms):
        self.mask = np.zeros(len(atoms), bool)
        self.mask[self.selection] = True
        constraints = atoms.constraints
        atoms.constraints = []
        self.qmatoms = atoms[self.mask]
        self.mmatoms = atoms[~self.mask]
        atoms.constraints = constraints
        self.qmatoms.pbc = False
        if self.vacuum:
            self.qmatoms.center(vacuum=self.vacuum)
            self.center = self.qmatoms.positions.mean(axis=0)
            print('Size of QM-cell after centering:', self.qmatoms.cell.diagonal(), file=self.output)
        self.qmatoms.calc = self.qmcalc
        self.mmatoms.calc = self.mmcalc
        if self.embedding is None:
            self.embedding = Embedding()
        self.embedding.initialize(self.qmatoms, self.mmatoms)
        print('Embedding:', self.embedding, file=self.output)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if self.qmatoms is None:
            self.initialize(atoms)
        self.mmatoms.set_positions(atoms.positions[~self.mask])
        self.qmatoms.set_positions(atoms.positions[self.mask])
        if self.vacuum:
            shift = self.center - self.qmatoms.positions.mean(axis=0)
            self.qmatoms.positions += shift
        else:
            shift = (0, 0, 0)
        self.embedding.update(shift)
        ienergy, iqmforces, immforces = self.interaction.calculate(self.qmatoms, self.mmatoms, shift)
        qmenergy = self.qmatoms.get_potential_energy()
        mmenergy = self.mmatoms.get_potential_energy()
        energy = ienergy + qmenergy + mmenergy
        print('Energies: {0:12.3f} {1:+12.3f} {2:+12.3f} = {3:12.3f}'.format(ienergy, qmenergy, mmenergy, energy), file=self.output)
        qmforces = self.qmatoms.get_forces()
        mmforces = self.mmatoms.get_forces()
        mmforces += self.embedding.get_mm_forces()
        forces = np.empty((len(atoms), 3))
        forces[self.mask] = qmforces + iqmforces
        forces[~self.mask] = mmforces + immforces
        self.results['energy'] = energy
        self.results['forces'] = forces