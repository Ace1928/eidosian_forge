import testtools
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils
from taskflow.types import failure
from taskflow.utils import eventlet_utils as eu
class FailingRetry(retry.Retry):

    def execute(self, **kwargs):
        raise ValueError('OMG I FAILED')

    def revert(self, history, **kwargs):
        self.history = history

    def on_failure(self, **kwargs):
        return retry.REVERT