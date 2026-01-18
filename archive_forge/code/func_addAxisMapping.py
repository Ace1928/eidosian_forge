from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
def addAxisMapping(self, axisMappingDescriptor: AxisMappingDescriptor):
    """Add the given ``axisMappingDescriptor`` to :attr:`axisMappings`."""
    self.axisMappings.append(axisMappingDescriptor)