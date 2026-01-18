from __future__ import annotations
import logging
import re
from itertools import chain, combinations
from typing import TYPE_CHECKING, no_type_check, overload
import numpy as np
from monty.fractions import gcd_float
from monty.json import MontyDecoder, MSONable
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
class ReactionError(Exception):
    """
    Exception class for Reactions. Allows more information in exception
    messages to cover situations not covered by standard exception classes.
    """

    def __init__(self, msg: str) -> None:
        """
        Create a ReactionError.

        Args:
            msg (str): More information about the ReactionError.
        """
        self.msg = msg

    def __str__(self) -> str:
        return self.msg