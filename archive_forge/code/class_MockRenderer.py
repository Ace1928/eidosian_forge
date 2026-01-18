import gc
from holoviews.core.element import Element
from holoviews.core.options import Keywords, Options, OptionTree, Store
from holoviews.core.spaces import HoloMap
from ..utils import LoggingComparisonTestCase
class MockRenderer:

    def __init__(self, backend):
        self.backend = backend