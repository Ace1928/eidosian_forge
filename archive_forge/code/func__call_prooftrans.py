import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
def _call_prooftrans(self, input_str, args=[], verbose=False):
    """
        Call the ``prooftrans`` binary with the given input.

        :param input_str: A string whose contents are used as stdin.
        :param args: A list of command-line arguments.
        :return: A tuple (stdout, returncode)
        :see: ``config_prover9``
        """
    if self._prooftrans_bin is None:
        self._prooftrans_bin = self._find_binary('prooftrans', verbose)
    return self._call(input_str, self._prooftrans_bin, args, verbose)