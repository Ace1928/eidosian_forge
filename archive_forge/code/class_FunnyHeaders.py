from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
class FunnyHeaders(Headers):
    pass