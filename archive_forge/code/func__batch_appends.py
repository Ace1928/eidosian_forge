from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
def _batch_appends(self, items):
    save = self.save
    write = self.write
    if not self.bin:
        for x in items:
            save(x)
            write(APPEND)
        return
    it = iter(items)
    while True:
        tmp = list(islice(it, self._BATCHSIZE))
        n = len(tmp)
        if n > 1:
            write(MARK)
            for x in tmp:
                save(x)
            write(APPENDS)
        elif n:
            save(tmp[0])
            write(APPEND)
        if n < self._BATCHSIZE:
            return