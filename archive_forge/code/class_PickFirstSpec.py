import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
class PickFirstSpec(nib.TraitedSpec):
    in_files = traits.List(File(exists=True), argstr='%s', position=2, mandatory=True)