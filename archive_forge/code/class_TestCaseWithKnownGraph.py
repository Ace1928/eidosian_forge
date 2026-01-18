import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
class TestCaseWithKnownGraph(tests.TestCase):
    scenarios = caching_scenarios()
    module = None

    def make_known_graph(self, ancestry):
        return self.module.KnownGraph(ancestry, do_cache=self.do_cache)