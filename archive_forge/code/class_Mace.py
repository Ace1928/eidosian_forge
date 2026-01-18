import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar
class Mace(Prover9Parent, ModelBuilder):
    _mace4_bin = None

    def __init__(self, end_size=500):
        self._end_size = end_size
        'The maximum model size that Mace will try before\n           simply returning false. (Use -1 for no maximum.)'

    def _build_model(self, goal=None, assumptions=None, verbose=False):
        """
        Use Mace4 to build a first order model.

        :return: ``True`` if a model was found (i.e. Mace returns value of 0),
        else ``False``
        """
        if not assumptions:
            assumptions = []
        stdout, returncode = self._call_mace4(self.prover9_input(goal, assumptions), verbose=verbose)
        return (returncode == 0, stdout)

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