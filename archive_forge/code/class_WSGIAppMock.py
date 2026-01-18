import sys
import pytest
from cheroot.cli import (
class WSGIAppMock:
    """Mock of a wsgi module."""

    def application(self):
        """Empty application method.

            Default method to be called when no specific callable
            is defined in the wsgi application identifier.

            It has an empty body because we are expecting to verify that
            the same method is return no the actual execution of it.
            """

    def main(self):
        """Empty custom method (callable) inside the mocked WSGI app.

            It has an empty body because we are expecting to verify that
            the same method is return no the actual execution of it.
            """