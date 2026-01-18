from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def _test_complement_none(complement_impl):
    now = datetime.now()
    ta = [['a', 'b'], [None, None]]
    tb = [['a', 'b'], [None, now]]
    expectation = (('a', 'b'), (None, None))
    result = complement_impl(ta, tb)
    ieq(expectation, result)
    ta = [['a'], [now], [None]]
    tb = [['a'], [None], [None]]
    expectation = (('a',), (now,))
    result = complement_impl(ta, tb)
    ieq(expectation, result)