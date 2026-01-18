import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
class ProductOutputSpec(nib.TraitedSpec):
    output1 = nib.traits.Int(mandatory=True, desc='output')