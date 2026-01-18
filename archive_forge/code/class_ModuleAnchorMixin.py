import unittest
import os
import contextlib
import importlib_resources as resources
class ModuleAnchorMixin:
    from . import data01 as anchor01
    from . import data02 as anchor02