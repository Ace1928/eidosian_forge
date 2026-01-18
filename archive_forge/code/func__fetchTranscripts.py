import re
import unittest
from typing import (
from . import (
def _fetchTranscripts(self) -> None:
    self.transcripts = {}
    testfiles = cast(List[str], getattr(self.cmdapp, 'testfiles', []))
    for fname in testfiles:
        tfile = open(fname)
        self.transcripts[fname] = iter(tfile.readlines())
        tfile.close()