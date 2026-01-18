import unittest
import cupy.testing._parameterized
@pytest.fixture(autouse=True)
def _cupy_testing_parameterize(self, _cupy_testing_param):
    assert not self.__dict__, 'There should not be another hack with instance attribute.'
    self.__dict__.update(_cupy_testing_param)