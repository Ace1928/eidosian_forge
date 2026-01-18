from io import StringIO
import numbers
from typing import (
from Bio import BiopythonDeprecationWarning, StreamModeError
from Bio.Seq import Seq, MutableSeq, UndefinedSequenceError
import warnings
def _set_per_letter_annotations(self, value: Mapping[str, str]) -> None:
    if not isinstance(value, dict):
        raise TypeError('The per-letter-annotations should be a (restricted) dictionary.')
    try:
        self._per_letter_annotations = _RestrictedDict(length=len(self.seq))
    except AttributeError:
        self._per_letter_annotations = _RestrictedDict(length=0)
    self._per_letter_annotations.update(value)