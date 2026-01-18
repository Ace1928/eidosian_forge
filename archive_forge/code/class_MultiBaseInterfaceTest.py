import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
class MultiBaseInterfaceTest(GeomTests):
    datatype = 'multitabular'
    interface = MultiInterface
    subtype = None
    __test__ = False

    def setUp(self):
        logger = get_logger()
        self._log_level = logger.level
        get_logger().setLevel(logging.ERROR)
        self._subtypes = MultiInterface.subtypes
        MultiInterface.subtypes = [self.subtype]
        super().setUp()

    def tearDown(self):
        MultiInterface.subtypes = self._subtypes
        get_logger().setLevel(self._log_level)
        super().tearDown()