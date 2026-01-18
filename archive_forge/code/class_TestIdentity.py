from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
class TestIdentity(SimplestIdentity, bakery.ACLIdentity):

    def allow(other, ctx, acls):
        self.assertEqual(ctx.get('a'), 'aval')
        Visited.in_allow = True
        return False