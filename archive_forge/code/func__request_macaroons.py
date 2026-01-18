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
def _request_macaroons(self, svc):
    mmap = self._macaroons.get(svc.name(), [])
    names = []
    for name in mmap:
        names.append(name)
    names = sorted(names)
    ms = [None] * len(names)
    for i, name in enumerate(names):
        ms[i] = mmap[name]
    return ms