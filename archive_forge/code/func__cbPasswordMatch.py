import os
from typing import Any, Dict, Optional, Tuple, Union
from zope.interface import Attribute, Interface, implementer
from twisted.cred import error
from twisted.cred.credentials import (
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import failure
def _cbPasswordMatch(self, matched, username):
    if matched:
        return username
    else:
        return failure.Failure(error.UnauthorizedLogin())