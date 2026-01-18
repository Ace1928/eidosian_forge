from taskflow import engines
from taskflow import formatters
from taskflow.listeners import logging as logging_listener
from taskflow.patterns import linear_flow
from taskflow import states
from taskflow import test
from taskflow.test import mock
from taskflow.test import utils as test_utils
def _make_test_flow(self):
    b = test_utils.TaskWithFailure('Broken')
    h_1 = test_utils.ProgressingTask('Happy-1')
    h_2 = test_utils.ProgressingTask('Happy-2')
    flo = linear_flow.Flow('test')
    flo.add(h_1, h_2, b)
    return flo