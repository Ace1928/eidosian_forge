import signal
import time
import testtools
from testtools.testcase import (
from testtools.matchers import raises
import fixtures
class GotAlarm(Exception):
    pass