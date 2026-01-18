from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import threading
import six
from six.moves import queue as Queue
from six.moves import range
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.seek_ahead_thread import SeekAheadResult
from gslib.seek_ahead_thread import SeekAheadThread
import gslib.tests.testcase as testcase
from gslib.ui_controller import UIController
from gslib.ui_controller import UIThread
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
from gslib.utils import unit_util
class SeekAheadResultIterator(object):

    def __init__(self, num_results):
        self.num_results = num_results
        self.yielded = 0

    def __iter__(self):
        while self.yielded < self.num_results:
            yield SeekAheadResult()
            self.yielded += 1