import unittest
import cupy.testing._parameterized
class _TestingParameterizeMixin:

    def __repr__(self):
        return '<{}  parameter: {}>'.format(super().__repr__(), self.__dict__)

    @pytest.fixture(autouse=True)
    def _cupy_testing_parameterize(self, _cupy_testing_param):
        assert not self.__dict__, 'There should not be another hack with instance attribute.'
        self.__dict__.update(_cupy_testing_param)