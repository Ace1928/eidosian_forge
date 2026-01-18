import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
def _vaspinput_factory(atoms=None, **kwargs) -> GenerateVaspInput:
    if atoms is None:
        atoms = nacl
    mocker = mock.Mock()
    inputs = GenerateVaspInput()
    inputs.set(**kwargs)
    inputs._build_pp_list = mocker(return_value=None)
    inputs.initialize(atoms)
    return inputs