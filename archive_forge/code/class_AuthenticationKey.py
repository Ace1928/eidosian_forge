import sys
from multiprocessing.context import assert_spawning
from multiprocessing.process import BaseProcess
class AuthenticationKey(bytes):

    def __reduce__(self):
        try:
            assert_spawning(self)
        except RuntimeError:
            raise TypeError('Pickling an AuthenticationKey object is disallowed for security reasons')
        return (AuthenticationKey, (bytes(self),))