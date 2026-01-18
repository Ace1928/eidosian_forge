import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
class FakeType(UserType):
    name = 'fake-type'
    basetype = int

    def validate(self, value):
        if value < 10:
            raise ValueError('should be greater than 10')

    def frombasetype(self, value):
        self.validate(value)