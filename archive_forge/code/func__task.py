from taskflow import exceptions as exc
from taskflow.patterns import graph_flow as gf
from taskflow import retry
from taskflow import test
from taskflow.tests import utils
def _task(name, provides=None, requires=None):
    return utils.ProvidesRequiresTask(name, provides, requires)