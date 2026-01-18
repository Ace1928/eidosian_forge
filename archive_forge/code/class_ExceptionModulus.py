import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (create_string_buffer,
from Cryptodome.Math._IntegerCustom import _raw_montgomery
class ExceptionModulus(ValueError):
    pass