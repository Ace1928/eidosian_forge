import pytest
from unittest import mock
import numpy as np
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import _args_without_comment
from ase.calculators.vasp.create_input import _to_vasp_bool, _from_vasp_bool
from ase.build import bulk
def _assert_magmom_equal_to_incar_value(atoms, expected_magmom, vaspinput):
    assert len(atoms) == len(expected_magmom)
    vaspinput.write_incar(atoms)
    new_magmom = read_magmom_from_file('INCAR')
    assert len(new_magmom) == len(expected_magmom)
    srt = vaspinput.sort
    resort = vaspinput.resort
    assert np.allclose(expected_magmom, new_magmom[resort], atol=0.001)
    assert np.allclose(np.array(expected_magmom)[srt], new_magmom, atol=0.001)