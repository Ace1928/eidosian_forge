import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
class ConfigureSubplotsBase(ToolBase):
    """Base tool for the configuration of subplots."""
    description = 'Configure subplots'
    image = 'subplots'