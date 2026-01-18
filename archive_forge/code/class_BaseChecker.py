import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
class BaseChecker:
    """Base class for retry checkers.

    Each class is responsible for checking a single criteria that determines
    whether or not a retry should not happen.

    """

    def __call__(self, attempt_number, response, caught_exception):
        """Determine if retry criteria matches.

        Note that either ``response`` is not None and ``caught_exception`` is
        None or ``response`` is None and ``caught_exception`` is not None.

        :type attempt_number: int
        :param attempt_number: The total number of times we've attempted
            to send the request.

        :param response: The HTTP response (if one was received).

        :type caught_exception: Exception
        :param caught_exception: Any exception that was caught while trying to
            send the HTTP response.

        :return: True, if the retry criteria matches (and therefore a retry
            should occur.  False if the criteria does not match.

        """
        if response is not None:
            return self._check_response(attempt_number, response)
        elif caught_exception is not None:
            return self._check_caught_exception(attempt_number, caught_exception)
        else:
            raise ValueError('Both response and caught_exception are None.')

    def _check_response(self, attempt_number, response):
        pass

    def _check_caught_exception(self, attempt_number, caught_exception):
        pass