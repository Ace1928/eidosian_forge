import numpy as np
import warnings
from ase.md.nvtberendsen import NVTBerendsen
import ase.units as units
def _process_pressure(self, pressure, pressure_au):
    """Handle that pressure can be specified in multiple units.

        For at least a transition period, Berendsen NPT dynamics in ASE can
        have the pressure specified in either bar or atomic units (eV/Å^3).

        Two parameters:

        pressure: None or float
            The original pressure specification in bar.
            A warning is issued if this is not None.

        pressure_au: None or float
            Pressure in ev/Å^3.

        Exactly one of the two pressure parameters must be different from 
        None, otherwise an error is issued.

        Return value: Pressure in eV/Å^3.
        """
    if (pressure is not None) + (pressure_au is not None) != 1:
        raise TypeError("Exactly one of the parameters 'pressure'," + " and 'pressure_au' must" + ' be given')
    if pressure is not None:
        w = "The 'pressure' parameter is deprecated, please" + ' specify the pressure in atomic units (eV/Å^3)' + " using the 'pressure_au' parameter."
        warnings.warn(FutureWarning(w))
        return pressure * (100000.0 * units.Pascal)
    else:
        return pressure_au