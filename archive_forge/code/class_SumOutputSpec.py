import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
class SumOutputSpec(nib.TraitedSpec):
    output1 = nib.traits.Int(desc='output')
    operands = nib.traits.List(nib.traits.Int, desc='operands')