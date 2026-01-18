import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
class CausedByTest(test_base.BaseTestCase):

    def test_caused_by_explicit(self):
        e = self.assertRaises(Fail1, excutils.raise_with_cause, Fail1, 'I was broken', cause=Fail2('I have been broken'))
        self.assertIsInstance(e.cause, Fail2)
        e_p = e.pformat()
        self.assertIn('I have been broken', e_p)
        self.assertIn('Fail2', e_p)

    def test_caused_by_implicit(self):

        def raises_chained():
            try:
                raise Fail2('I have been broken')
            except Fail2:
                excutils.raise_with_cause(Fail1, 'I was broken')
        e = self.assertRaises(Fail1, raises_chained)
        self.assertIsInstance(e.cause, Fail2)
        e_p = e.pformat()
        self.assertIn('I have been broken', e_p)
        self.assertIn('Fail2', e_p)