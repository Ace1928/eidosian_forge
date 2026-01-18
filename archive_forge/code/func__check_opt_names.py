import stevedore
from testtools import matchers
from glance_store import backend
from glance_store.tests import base
def _check_opt_names(self, opt_list, expected_opt_names):
    opt_names = [o.name for g, l in opt_list for o in l]
    self.assertThat(opt_names, matchers.HasLength(len(expected_opt_names)))
    for opt in opt_names:
        self.assertIn(opt, expected_opt_names)