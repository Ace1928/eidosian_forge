import testtools
from neutronclient.common import exceptions
from neutronclient.common import validators
def _test_validate_int(self, attr_val, attr_name='attr1', min_value=1, max_value=10):
    obj = FakeParsedArgs()
    setattr(obj, attr_name, attr_val)
    ret = validators.validate_int_range(obj, attr_name, min_value, max_value)
    self.assertIsNone(ret)