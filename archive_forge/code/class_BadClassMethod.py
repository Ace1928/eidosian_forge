import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
class BadClassMethod:
    not_callable = classmethod(None)