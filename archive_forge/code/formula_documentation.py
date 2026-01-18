from math import gcd
import re
from typing import Dict, Tuple, List, Sequence, Union
from ase.data import chemical_symbols, atomic_numbers
Return the tuple (self // other, self % other).

        Invariant::

            div, mod = divmod(self, other)
            div * other + mod == self

        Example
        -------
        >>> divmod(Formula('H2O'), 'H')
        (2, Formula('O'))
        