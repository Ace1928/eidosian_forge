import logging
from io import StringIO
from operator import itemgetter, attrgetter
from pyomo.common.config import (
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.label import LPFileLabeler, NumericLabeler
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import (
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port
class LPWriterInfo(object):
    """Return type for LPWriter.write()

    Attributes
    ----------
    symbol_map: SymbolMap

        The :py:class:`SymbolMap` bimap between row/column labels and
        Pyomo components.

    """

    def __init__(self, symbol_map):
        self.symbol_map = symbol_map