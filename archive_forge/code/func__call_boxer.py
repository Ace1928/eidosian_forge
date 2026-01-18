import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
def _call_boxer(self, candc_out, verbose=False):
    """
        Call the ``boxer`` binary with the given input.

        :param candc_out: str output from C&C parser
        :return: stdout
        """
    f = None
    try:
        fd, temp_filename = tempfile.mkstemp(prefix='boxer-', suffix='.in', text=True)
        f = os.fdopen(fd, 'w')
        f.write(candc_out.decode('utf-8'))
    finally:
        if f:
            f.close()
    args = ['--box', 'false', '--semantics', 'drs', '--resolve', ['false', 'true'][self._resolve], '--elimeq', ['false', 'true'][self._elimeq], '--format', 'prolog', '--instantiate', 'true', '--input', temp_filename]
    stdout = self._call(None, self._boxer_bin, args, verbose)
    os.remove(temp_filename)
    return stdout