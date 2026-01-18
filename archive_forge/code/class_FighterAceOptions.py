import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
class FighterAceOptions(usage.Options):
    """
    Command-line options for an imaginary `Fighter Ace` game
    """
    optFlags: List[List[Optional[str]]] = [['fokker', 'f', 'Select the Fokker Dr.I as your dogfighter aircraft'], ['albatros', 'a', 'Select the Albatros D-III as your dogfighter aircraft'], ['spad', 's', 'Select the SPAD S.VII as your dogfighter aircraft'], ['bristol', 'b', 'Select the Bristol Scout as your dogfighter aircraft'], ['physics', 'p', 'Enable secret Twisted physics engine'], ['jam', 'j', 'Enable a small chance that your machine guns will jam!'], ['verbose', 'v', 'Verbose logging (may be specified more than once)']]
    optParameters: List[List[Optional[str]]] = [['pilot-name', None, "What's your name, Ace?", 'Manfred von Richthofen'], ['detail', 'd', 'Select the level of rendering detail (1-5)', '3']]
    subCommands = [['server', None, FighterAceServerOptions, 'Start FighterAce game-server.']]
    compData = Completions(descriptions={'physics': 'Twisted-Physics', 'detail': 'Rendering detail level'}, multiUse=['verbose'], mutuallyExclusive=[['fokker', 'albatros', 'spad', 'bristol']], optActions={'detail': CompleteList(['12345'])}, extraActions=[CompleteFiles(descr='saved game file to load')])

    def opt_silly(self):
        """ """