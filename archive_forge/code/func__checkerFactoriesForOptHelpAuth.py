import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
def _checkerFactoriesForOptHelpAuth(self):
    """
        Return a list of which authTypes will be displayed by --help-auth.
        This makes it a lot easier to test this module.
        """
    for factory in findCheckerFactories():
        for interface in factory.credentialInterfaces:
            if self.supportsInterface(interface):
                yield factory
                break