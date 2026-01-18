import os
import sys
from contextlib import ExitStack
import numpy as np
from ase.db.core import Database, ops, lock, now
from ase.db.row import AtomsRow
from ase.io.jsonio import encode, decode
from ase.parallel import world, parallel_function
def _write_json(self, bigdct, ids, nextid):
    if world.rank > 0:
        return
    with ExitStack() as stack:
        if isinstance(self.filename, str):
            fd = stack.enter_context(open(self.filename, 'w'))
        else:
            fd = self.filename
        print('{', end='', file=fd)
        for id in ids:
            dct = bigdct[id]
            txt = ',\n '.join(('"{0}": {1}'.format(key, encode(dct[key])) for key in sorted(dct.keys())))
            print('"{0}": {{\n {1}}},'.format(id, txt), file=fd)
        if self._metadata is not None:
            print('"metadata": {0},'.format(encode(self.metadata)), file=fd)
        print('"ids": {0},'.format(ids), file=fd)
        print('"nextid": {0}}}'.format(nextid), file=fd)