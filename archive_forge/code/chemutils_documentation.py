import os
import re
from rdkit import RDConfig
 counts the number of electrons appearing in a configuration string

      **Arguments**

        - config: the configuration string (e.g. '2s^2 2p^4')

        - ignoreFullD: toggles not counting full d shells

        - ignoreFullF: toggles not counting full f shells

      **Returns**

        the number of valence electrons

    