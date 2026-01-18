import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
def addADF(self, adfset):
    """Add an Automatically Defined Function (ADF) to the set.

        :param adfset: PrimitiveSetTyped containing the primitives with which
                       the ADF can be built.
        """
    prim = Primitive(adfset.name, adfset.ins, adfset.ret)
    self._add(prim)
    self.prims_count += 1