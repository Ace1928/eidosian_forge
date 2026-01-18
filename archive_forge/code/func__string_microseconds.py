from datetime import (
import time
import unittest
def _string_microseconds(d, timezone):
    return '%04d-%02d-%02dT%02d:%02d:%02d.%06d%s' % (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, timezone)