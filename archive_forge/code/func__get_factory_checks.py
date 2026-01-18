import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def _get_factory_checks(self, factory):
    check_fns = []

    def _reg(check_fn):
        self.assertTrue(hasattr(check_fn, '__call__'))
        self.assertNotIn(check_fn, check_fns)
        check_fns.append(check_fn)
    factory(_reg)
    return check_fns