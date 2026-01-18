from taskflow import exceptions
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow.patterns import unordered_flow as uf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
class FullArgsRetry(retry.AlwaysRevert):

    def execute(self, history, **kwargs):
        pass

    def revert(self, history, **kwargs):
        pass