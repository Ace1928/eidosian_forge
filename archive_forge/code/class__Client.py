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
class _Client(object):
    max_retries = 3

    def __init__(self, dischargers):
        self._key = bakery.generate_key()
        self._macaroons = {}
        self._dischargers = dischargers

    def do(self, ctx, svc, ops):

        class _AuthInfo:
            authInfo = None

        def svc_do(ms):
            _AuthInfo.authInfo = svc.do(ctx, ms, ops)
        self._do_func(ctx, svc, svc_do)
        return _AuthInfo.authInfo

    def do_any(self, ctx, svc, ops):
        return svc.do_any(ctx, self._request_macaroons(svc), ops)

    def capability(self, ctx, svc, ops):

        class _M:
            m = None

        def svc_capability(ms):
            _M.m = svc.capability(ctx, ms, ops)
            return
        self._do_func(ctx, svc, svc_capability)
        return _M.m

    def discharged_capability(self, ctx, svc, ops):
        m = self.capability(ctx, svc, ops)
        return self._discharge_all(ctx, m)

    def _do_func(self, ctx, svc, f):
        for i in range(0, self.max_retries):
            try:
                f(self._request_macaroons(svc))
                return
            except _DischargeRequiredError as exc:
                ms = self._discharge_all(ctx, exc.m())
                self.add_macaroon(svc, exc.name(), ms)
        raise ValueError('discharge failed too many times')

    def _clear_macaroons(self, svc):
        if svc is None:
            self._macaroons = {}
            return
        if svc.name() in self._macaroons:
            del self._macaroons[svc.name()]

    def add_macaroon(self, svc, name, m):
        if svc.name() not in self._macaroons:
            self._macaroons[svc.name()] = {}
        self._macaroons[svc.name()][name] = m

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

    def _discharge_all(self, ctx, m):

        def get_discharge(cav, payload):
            d = self._dischargers.get(cav.location)
            if d is None:
                raise ValueError('third party discharger {} not found'.format(cav.location))
            return d.discharge(ctx, cav, payload)
        return bakery.discharge_all(m, get_discharge)