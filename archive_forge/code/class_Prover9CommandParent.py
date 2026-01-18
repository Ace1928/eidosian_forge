import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
class Prover9CommandParent:
    """
    A common base class used by both ``Prover9Command`` and ``MaceCommand``,
    which is responsible for maintaining a goal and a set of assumptions,
    and generating prover9-style input files from them.
    """

    def print_assumptions(self, output_format='nltk'):
        """
        Print the list of the current assumptions.
        """
        if output_format.lower() == 'nltk':
            for a in self.assumptions():
                print(a)
        elif output_format.lower() == 'prover9':
            for a in convert_to_prover9(self.assumptions()):
                print(a)
        else:
            raise NameError("Unrecognized value for 'output_format': %s" % output_format)