import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class ManilaParseException(Exception):
    """The base exception class for all exceptions this library raises."""

    def __init__(self, message=None, details=None):
        self.message = message or 'Argument parse exception'
        self.details = details or None

    def __str__(self):
        return self.message