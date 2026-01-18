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
def _do_func(self, ctx, svc, f):
    for i in range(0, self.max_retries):
        try:
            f(self._request_macaroons(svc))
            return
        except _DischargeRequiredError as exc:
            ms = self._discharge_all(ctx, exc.m())
            self.add_macaroon(svc, exc.name(), ms)
    raise ValueError('discharge failed too many times')