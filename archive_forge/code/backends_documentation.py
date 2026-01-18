import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
Allow this object to be serialized with pickle.

        This uses the global registry `_registered_algorithms` to deserialize.
        