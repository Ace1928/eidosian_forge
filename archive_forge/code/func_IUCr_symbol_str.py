from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
@property
def IUCr_symbol_str(self):
    """Returns a string representation of the IUCr symbol of this coordination geometry."""
    return str(self.IUCrsymbol)