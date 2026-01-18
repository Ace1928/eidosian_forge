import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
class DescriptionTest(common.HeatTestCase):

    def setUp(self):
        super(DescriptionTest, self).setUp()

    def test_func(self):

        def f():
            pass
        self.assertEqual('f', scheduler.task_description(f))

    def test_lambda(self):
        lam = lambda: None
        self.assertEqual('<lambda>', scheduler.task_description(lam))

    def test_method(self):

        class C(object):

            def __str__(self):
                return 'C "o"'

            def __repr__(self):
                return 'o'

            def m(self):
                pass
        self.assertEqual('m from C "o"', scheduler.task_description(C().m))

    def test_object(self):

        class C(object):

            def __str__(self):
                return 'C "o"'

            def __repr__(self):
                return 'o'

            def __call__(self):
                pass
        self.assertEqual('o', scheduler.task_description(C()))

    def test_unicode(self):

        class C(object):

            def __str__(self):
                return u'C "♥"'

            def __repr__(self):
                return u'♥'

            def __call__(self):
                pass

            def m(self):
                pass
        self.assertEqual(u'm from C "♥"', scheduler.task_description(C().m))
        self.assertEqual(u'♥', scheduler.task_description(C()))