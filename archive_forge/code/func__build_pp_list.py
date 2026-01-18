import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def _build_pp_list(self, atoms, setups=None, special_setups: Sequence[int]=()):
    """Build the pseudopotential lists"""
    p = self.input_params
    if setups is None:
        setups, special_setups = self._get_setups()
    symbols, _ = count_symbols(atoms, exclude=special_setups)
    for pp_alias, pp_folder in (('lda', 'potpaw'), ('pw91', 'potpaw_GGA'), ('pbe', 'potpaw_PBE')):
        if p['pp'].lower() == pp_alias:
            break
    else:
        pp_folder = p['pp']
    if self.VASP_PP_PATH in os.environ:
        pppaths = os.environ[self.VASP_PP_PATH].split(':')
    else:
        pppaths = []
    ppp_list = []
    for m in special_setups:
        if m in setups:
            special_setup_index = m
        elif str(m) in setups:
            special_setup_index = str(m)
        else:
            raise Exception('Having trouble with special setup index {0}. Please use an int.'.format(m))
        potcar = join(pp_folder, setups[special_setup_index], 'POTCAR')
        for path in pppaths:
            filename = join(path, potcar)
            if isfile(filename) or islink(filename):
                ppp_list.append(filename)
                break
            elif isfile(filename + '.Z') or islink(filename + '.Z'):
                ppp_list.append(filename + '.Z')
                break
        else:
            symbol = atoms.symbols[m]
            msg = 'Looking for {}.\n                No pseudopotential for symbol{} with setup {} '.format(potcar, symbol, setups[special_setup_index])
            raise RuntimeError(msg)
    for symbol in symbols:
        try:
            potcar = join(pp_folder, symbol + setups[symbol], 'POTCAR')
        except (TypeError, KeyError):
            potcar = join(pp_folder, symbol, 'POTCAR')
        for path in pppaths:
            filename = join(path, potcar)
            if isfile(filename) or islink(filename):
                ppp_list.append(filename)
                break
            elif isfile(filename + '.Z') or islink(filename + '.Z'):
                ppp_list.append(filename + '.Z')
                break
        else:
            msg = 'Looking for PP for {}\n                        The pseudopotentials are expected to be in:\n                        LDA:  $VASP_PP_PATH/potpaw/\n                        PBE:  $VASP_PP_PATH/potpaw_PBE/\n                        PW91: $VASP_PP_PATH/potpaw_GGA/\n                        \n                        No pseudopotential for {}!'.format(potcar, symbol)
            raise RuntimeError(msg)
    return ppp_list