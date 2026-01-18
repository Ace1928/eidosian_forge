from __future__ import annotations
import abc
import copy
import json
import logging
import os
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.collections import AttrDict
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.abinit import abiobjects as aobj
from pymatgen.io.abinit.pseudos import Pseudo, PseudoTable
from pymatgen.io.abinit.variable import InputVariable
from pymatgen.symmetry.bandstructure import HighSymmKpath
def _find_ecut_pawecutdg(ecut, pawecutdg, pseudos, accuracy):
    """Return a |AttrDict| with the value of ecut and pawecutdg."""
    if ecut is None or (pawecutdg is None and any((p.ispaw for p in pseudos))):
        has_hints = all((p.has_hints for p in pseudos))
    if ecut is None:
        if has_hints:
            ecut = max((p.hint_for_accuracy(accuracy).ecut for p in pseudos))
        else:
            raise RuntimeError('ecut is None but pseudos do not provide hints for ecut')
    if pawecutdg is None and any((p.ispaw for p in pseudos)):
        if has_hints:
            pawecutdg = max((p.hint_for_accuracy(accuracy).pawecutdg for p in pseudos))
        else:
            raise RuntimeError('pawecutdg is None but pseudos do not provide hints')
    return AttrDict(ecut=ecut, pawecutdg=pawecutdg)