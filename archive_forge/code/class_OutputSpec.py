import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
class OutputSpec(nib.TraitedSpec):
    files = nib.traits.List(nib.File)
    others = nib.File()