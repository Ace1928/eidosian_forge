import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar
def _call_interpformat(self, input_str, args=[], verbose=False):
    """
        Call the ``interpformat`` binary with the given input.

        :param input_str: A string whose contents are used as stdin.
        :param args: A list of command-line arguments.
        :return: A tuple (stdout, returncode)
        :see: ``config_prover9``
        """
    if self._interpformat_bin is None:
        self._interpformat_bin = self._modelbuilder._find_binary('interpformat', verbose)
    return self._modelbuilder._call(input_str, self._interpformat_bin, args, verbose)