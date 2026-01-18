import unittest
import os
import contextlib
import importlib_resources as resources
class FunctionalAPITest_ModuleAnchor(unittest.TestCase, FunctionalAPIBase, ModuleAnchorMixin):
    pass