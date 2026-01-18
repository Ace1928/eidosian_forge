import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
class MultiChecker(BaseChecker):

    def __init__(self, checkers):
        self._checkers = checkers

    def __call__(self, attempt_number, response, caught_exception):
        for checker in self._checkers:
            checker_response = checker(attempt_number, response, caught_exception)
            if checker_response:
                return checker_response
        return False