from __future__ import annotations
import os
import shutil
import subprocess
from glob import glob
from typing import TYPE_CHECKING, Literal
from urllib.request import urlretrieve
from monty.json import jsanitize
from monty.serialization import dumpfn, loadfn
from ruamel import yaml
from pymatgen.core import OLD_SETTINGS_FILE, SETTINGS_FILE, Element
from pymatgen.io.cp2k.inputs import GaussianTypeOrbitalBasisSet, GthPotential
from pymatgen.io.cp2k.utils import chunk
def add_config_var(tokens: list[str], backup_suffix: str) -> None:
    """Add/update keys in .pmgrc.yaml config file."""
    if len(tokens) % 2 != 0:
        raise ValueError(f'Uneven number {len(tokens)} of tokens passed to pmg config. Needs a value for every key.')
    if os.path.isfile(SETTINGS_FILE):
        rc_path = SETTINGS_FILE
    elif os.path.isfile(OLD_SETTINGS_FILE):
        rc_path = OLD_SETTINGS_FILE
    else:
        rc_path = SETTINGS_FILE
    dct = {}
    if os.path.isfile(rc_path):
        if backup_suffix:
            shutil.copy(rc_path, rc_path + backup_suffix)
            print(f'Existing {rc_path} backed up to {rc_path}{backup_suffix}')
        dct = loadfn(rc_path)
    special_vals = {'true': True, 'false': False, 'none': None, 'null': None}
    for key, val in zip(tokens[0::2], tokens[1::2]):
        dct[key] = special_vals.get(val.lower(), val)
    dumpfn(dct, rc_path)
    print(f'New {rc_path} written!')