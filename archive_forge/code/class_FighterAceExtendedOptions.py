import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
class FighterAceExtendedOptions(FighterAceOptions):
    """
    Extend the options and zsh metadata provided by FighterAceOptions.
    _shellcomp must accumulate options and metadata from all classes in the
    hiearchy so this is important to test.
    """
    optFlags = [['no-stalls', None, 'Turn off the ability to stall your aircraft']]
    optParameters = [['reality-level', None, 'Select the level of physics reality (1-5)', '5']]
    compData = Completions(descriptions={'no-stalls': "Can't stall your plane"}, optActions={'reality-level': Completer(descr='Physics reality level')})

    def opt_nocrash(self):
        """
        Select that you can't crash your plane
        """

    def opt_difficulty(self, difficulty):
        """
        How tough are you? (1-10)
        """