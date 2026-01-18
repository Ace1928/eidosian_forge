from typing import Dict, Any
import numpy as np
import ase
from ase.db import connect
from ase.calculators.calculator import Calculator
class CheckpointCalculator(Calculator):
    """
    This wraps any calculator object to checkpoint whenever a calculation
    is performed.

    This is particularly useful for expensive calculators, e.g. DFT and
    allows usage of complex workflows.

    Example usage:

        calc = ...
        cp_calc = CheckpointCalculator(calc)
        atoms.calc = cp_calc
        e = atoms.get_potential_energy()
        # 1st time, does calc, writes to checkfile
        # subsequent runs, reads from checkpoint file
    """
    implemented_properties = ase.calculators.calculator.all_properties
    default_parameters: Dict[str, Any] = {}
    name = 'CheckpointCalculator'
    property_to_method_name = {'energy': 'get_potential_energy', 'energies': 'get_potential_energies', 'forces': 'get_forces', 'stress': 'get_stress', 'stresses': 'get_stresses'}

    def __init__(self, calculator, db='checkpoints.db', logfile=None):
        Calculator.__init__(self)
        self.calculator = calculator
        if logfile is None:
            logfile = DevNull()
        self.checkpoint = Checkpoint(db, logfile)
        self.logfile = logfile

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        try:
            results = self.checkpoint.load(atoms)
            prev_atoms, results = (results[0], results[1:])
            try:
                assert atoms_almost_equal(atoms, prev_atoms)
            except AssertionError:
                raise AssertionError('mismatch between current atoms and those read from checkpoint file')
            self.logfile.write('retrieved results for {0} from checkpoint\n'.format(properties))
            if isinstance(self.calculator, Calculator):
                if not hasattr(self.calculator, 'results'):
                    self.calculator.results = {}
                self.calculator.results.update(dict(zip(properties, results)))
        except NoCheckpoint:
            if isinstance(self.calculator, Calculator):
                self.logfile.write('doing calculation of {0} with new-style calculator interface\n'.format(properties))
                self.calculator.calculate(atoms, properties, system_changes)
                results = [self.calculator.results[prop] for prop in properties]
            else:
                self.logfile.write('doing calculation of {0} with old-style calculator interface\n'.format(properties))
                results = []
                for prop in properties:
                    method_name = self.property_to_method_name[prop]
                    method = getattr(self.calculator, method_name)
                    results.append(method(atoms))
            _calculator = atoms.calc
            try:
                atoms.calc = self.calculator
                self.checkpoint.save(atoms, *results)
            finally:
                atoms.calc = _calculator
        self.results = dict(zip(properties, results))