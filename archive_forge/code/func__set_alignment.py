import subprocess
from collections import namedtuple
def _set_alignment(self, alignment):
    _check_alignment(len(self.words), len(self.mots), alignment)
    self._alignment = alignment