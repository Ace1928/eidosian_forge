import io
import logging
import sys
from collections.abc import Sequence
from typing import Optional, List, TextIO
from pyomo.common.config import (
from pyomo.common.log import LogStream
from pyomo.common.numeric_types import native_logical_types
from pyomo.common.timing import HierarchicalTimer
class PersistentSolverConfig(SolverConfig):
    """
    Base config for all persistent solver interfaces
    """

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        super().__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)
        self.auto_updates: AutoUpdateConfig = self.declare('auto_updates', AutoUpdateConfig())