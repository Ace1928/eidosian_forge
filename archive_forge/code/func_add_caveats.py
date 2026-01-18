import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons.verifier import Verifier
def add_caveats(m):
    for i in range(0, 1):
        if State.total_required == 0:
            break
        cid = 'id{}'.format(State.id)
        m.macaroon.add_third_party_caveat(location='somewhere', key='root key {}'.format(cid).encode('utf-8'), key_id=cid.encode('utf-8'))
        State.id += 1
        State.total_required -= 1