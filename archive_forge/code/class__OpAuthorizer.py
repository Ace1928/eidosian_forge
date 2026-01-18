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
class _OpAuthorizer(bakery.Authorizer):
    """Implements bakery.Authorizer by looking the operation
    up in the given map. If the username is in the associated list
    or the list contains "everyone", authorization is granted.
    """

    def __init__(self, auth=None):
        if auth is None:
            auth = {}
        self._auth = auth

    def authorize(self, ctx, id, ops):
        return bakery.ACLAuthorizer(allow_public=True, get_acl=lambda ctx, op: self._auth.get(op, [])).authorize(ctx, id, ops)