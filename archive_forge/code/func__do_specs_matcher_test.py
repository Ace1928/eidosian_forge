from oslotest import base as test_base
from oslo_utils import specs_matcher
def _do_specs_matcher_test(self, value, req, matches):
    assertion = self.assertTrue if matches else self.assertFalse
    assertion(specs_matcher.match(value, req))