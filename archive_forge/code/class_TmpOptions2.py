import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
class TmpOptions2(FighterAceExtendedOptions):
    compData = Completions(mutuallyExclusive=[('foo', 'bar')])