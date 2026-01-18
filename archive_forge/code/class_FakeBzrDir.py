import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
class FakeBzrDir:

    def __init__(self):
        self.calls = []

    def open_repository(self):
        self.calls.append('open_repository')
        raise errors.NoRepositoryPresent(real_bzrdir)