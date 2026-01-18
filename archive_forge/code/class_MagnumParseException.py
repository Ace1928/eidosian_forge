import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
class MagnumParseException(Exception):
    """The base exception class for all exceptions this library raises."""

    def __init__(self, message=None, details=None):
        self.message = message or 'Argument parse exception'
        self.details = details or None

    def __str__(self):
        return self.message