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
def _discharge_required_error(self, err):
    m = self._store.new_macaroon(err.cavs(), self._checker.namespace(), err.ops())
    name = 'authz'
    if len(err.ops()) == 1 and err.ops()[0] == bakery.LOGIN_OP:
        name = 'authn'
    raise _DischargeRequiredError(name=name, m=m)