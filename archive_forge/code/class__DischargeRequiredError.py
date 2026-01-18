import base64
import json
from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery.tests.common import epoch, test_checker, test_context
from pymacaroons.verifier import FirstPartyCaveatVerifierDelegate, Verifier
class _DischargeRequiredError(Exception):

    def __init__(self, name, m):
        Exception.__init__(self, 'discharge required')
        self._name = name
        self._m = m

    def m(self):
        return self._m

    def name(self):
        return self._name