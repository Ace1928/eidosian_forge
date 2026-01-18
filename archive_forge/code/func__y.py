from collections.abc import Iterable
import warnings
from typing import Sequence
def _y(self, wire):
    """Used for determining the correct y coordinate for classical wires.
        Classical wires should be enumerated starting at the number of quantum wires the drawer has.
        For example, if the drawer has ``3`` quantum wires, the first classical wire should be located at ``3``
        which corresponds to a ``y`` coordinate of ``2.9``.
        """
    if wire < self.n_wires:
        return wire
    return self.n_wires + self._cwire_scaling * (wire - self.n_wires)