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
class _Discharger(object):
    """ utility class that has a discharge function with the same signature of
    get_discharge for discharge_all.
    """

    def __init__(self, key, locator, checker):
        self._key = key
        self._locator = locator
        self._checker = checker

    def discharge(self, ctx, cav, payload):
        return bakery.discharge(ctx, key=self._key, id=cav.caveat_id, caveat=payload, checker=self._checker, locator=self._locator)