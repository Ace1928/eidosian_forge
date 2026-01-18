import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def _merge_tags(existing, changed):
    new_tags, gone_tags = changed
    result_new = set(existing[0])
    result_gone = set(existing[1])
    result_new.update(new_tags)
    result_new.difference_update(gone_tags)
    result_gone.update(gone_tags)
    result_gone.difference_update(new_tags)
    return (result_new, result_gone)