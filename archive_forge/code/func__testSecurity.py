import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def _testSecurity(self, inputList, atom):
    """
        Helper test method to test security options for a type.

        @param inputList: a sample input for the type.
        @type inputList: L{list}

        @param atom: atom identifier for the type.
        @type atom: L{str}
        """
    c = jelly.jelly(inputList)
    taster = jelly.SecurityOptions()
    taster.allowBasicTypes()
    jelly.unjelly(c, taster)
    taster.allowedTypes.pop(atom)
    self.assertRaises(jelly.InsecureJelly, jelly.unjelly, c, taster)