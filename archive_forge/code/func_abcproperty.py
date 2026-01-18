import numpy as np
from ase.data import atomic_numbers, chemical_symbols, atomic_masses
def abcproperty(index):
    """Helper function to easily create Atom ABC-property."""

    def getter(self):
        return self.scaled_position[index]

    def setter(self, value):
        spos = self.scaled_position
        spos[index] = value
        self.scaled_position = spos
    return property(getter, setter, doc='ABC'[index] + '-coordinate')