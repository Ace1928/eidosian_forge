import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar
def _call_mace4(self, input_str, args=[], verbose=False):
    """
        Call the ``mace4`` binary with the given input.

        :param input_str: A string whose contents are used as stdin.
        :param args: A list of command-line arguments.
        :return: A tuple (stdout, returncode)
        :see: ``config_prover9``
        """
    if self._mace4_bin is None:
        self._mace4_bin = self._find_binary('mace4', verbose)
    updated_input_str = ''
    if self._end_size > 0:
        updated_input_str += 'assign(end_size, %d).\n\n' % self._end_size
    updated_input_str += input_str
    return self._call(updated_input_str, self._mace4_bin, args, verbose)