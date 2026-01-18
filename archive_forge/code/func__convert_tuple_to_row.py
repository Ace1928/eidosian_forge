import json
import numbers
import os
import sqlite3
import sys
from contextlib import contextmanager
import numpy as np
import ase.io.jsonio
from ase.data import atomic_numbers
from ase.calculators.calculator import all_properties
from ase.db.row import AtomsRow
from ase.db.core import (Database, ops, now, lock, invop, parse_selection,
from ase.parallel import parallel_function
def _convert_tuple_to_row(self, values):
    deblob = self.deblob
    decode = self.decode
    values = self._old2new(values)
    dct = {'id': values[0], 'unique_id': values[1], 'ctime': values[2], 'mtime': values[3], 'user': values[4], 'numbers': deblob(values[5], np.int32), 'positions': deblob(values[6], shape=(-1, 3)), 'cell': deblob(values[7], shape=(3, 3))}
    if values[8] is not None:
        dct['pbc'] = (values[8] & np.array([1, 2, 4])).astype(bool)
    if values[9] is not None:
        dct['initial_magmoms'] = deblob(values[9])
    if values[10] is not None:
        dct['initial_charges'] = deblob(values[10])
    if values[11] is not None:
        dct['masses'] = deblob(values[11])
    if values[12] is not None:
        dct['tags'] = deblob(values[12], np.int32)
    if values[13] is not None:
        dct['momenta'] = deblob(values[13], shape=(-1, 3))
    if values[14] is not None:
        dct['constraints'] = values[14]
    if values[15] is not None:
        dct['calculator'] = values[15]
    if values[16] is not None:
        dct['calculator_parameters'] = decode(values[16])
    if values[17] is not None:
        dct['energy'] = values[17]
    if values[18] is not None:
        dct['free_energy'] = values[18]
    if values[19] is not None:
        dct['forces'] = deblob(values[19], shape=(-1, 3))
    if values[20] is not None:
        dct['stress'] = deblob(values[20])
    if values[21] is not None:
        dct['dipole'] = deblob(values[21])
    if values[22] is not None:
        dct['magmoms'] = deblob(values[22])
    if values[23] is not None:
        dct['magmom'] = values[23]
    if values[24] is not None:
        dct['charges'] = deblob(values[24])
    if values[25] != '{}':
        dct['key_value_pairs'] = decode(values[25])
    if len(values) >= 27 and values[26] != 'null':
        dct['data'] = decode(values[26], lazy=True)
    external_tab = self._get_external_table_names()
    tables = {}
    for tab in external_tab:
        row = self._read_external_table(tab, dct['id'])
        tables[tab] = row
    dct.update(tables)
    return AtomsRow(dct)