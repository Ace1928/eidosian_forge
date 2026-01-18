from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
class PawPseudo(abc.ABC):
    """
    Abstract class that defines the methods that must be implemented
    by the concrete classes representing PAW pseudopotentials.
    """

    @property
    @abc.abstractmethod
    def paw_radius(self):
        """Radius of the PAW sphere in a.u."""

    @property
    def rcore(self):
        """Alias of paw_radius."""
        return self.paw_radius