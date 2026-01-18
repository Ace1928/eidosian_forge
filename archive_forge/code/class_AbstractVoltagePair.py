from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MSONable
from scipy.constants import N_A
from pymatgen.core import Composition, Element
@dataclass
class AbstractVoltagePair(MSONable):
    """An Abstract Base Class for a Voltage Pair.

    Attributes:
        voltage : Voltage of voltage pair.
        mAh: Energy in mAh.
        mass_charge: Mass of charged pair.
        mass_discharge: Mass of discharged pair.
        vol_charge: Vol of charged pair.
        vol_discharge: Vol of discharged pair.
        frac_charge: Frac of working ion in charged pair.
        frac_discharge: Frac of working ion in discharged pair.
        working_ion_entry: Working ion as an entry.
        framework_formula : The compositions of one formula unit of the host material
    """
    voltage: float
    mAh: float
    mass_charge: float
    mass_discharge: float
    vol_charge: float
    vol_discharge: float
    frac_charge: float
    frac_discharge: float
    working_ion_entry: ComputedEntry
    framework_formula: str

    def __post_init__(self):
        fw = Composition(self.framework_formula)
        self.framework_formula = fw.reduced_formula

    @property
    def working_ion(self) -> Element:
        """Working ion as pymatgen Element object."""
        return self.working_ion_entry.elements[0]

    @property
    def framework(self) -> Composition:
        """The composition object representing the framework."""
        return Composition(self.framework_formula)

    @property
    def x_charge(self) -> float:
        """The number of working ions per formula unit of host in the charged state."""
        return self.frac_charge * self.framework.num_atoms / (1 - self.frac_charge)

    @property
    def x_discharge(self) -> float:
        """The number of working ions per formula unit of host in the discharged state."""
        return self.frac_discharge * self.framework.num_atoms / (1 - self.frac_discharge)