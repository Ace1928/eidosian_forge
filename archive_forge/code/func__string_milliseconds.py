from datetime import (
import time
import unittest
def _string_milliseconds(d, timezone):
    return '%04d-%02d-%02dT%02d:%02d:%02d.%03d%s' % (d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond / 1000, timezone)