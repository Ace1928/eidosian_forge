import os
import tempfile
from nltk.inference.api import BaseModelBuilderCommand, ModelBuilder
from nltk.inference.prover9 import Prover9CommandParent, Prover9Parent
from nltk.sem import Expression, Valuation
from nltk.sem.logic import is_indvar
def _transform_output(self, valuation_str, format):
    """
        Transform the output file into any Mace4 ``interpformat`` format.

        :param format: Output format for displaying models.
        :type format: str
        """
    if format in ['standard', 'standard2', 'portable', 'tabular', 'raw', 'cooked', 'xml', 'tex']:
        return self._call_interpformat(valuation_str, [format])[0]
    else:
        raise LookupError('The specified format does not exist')