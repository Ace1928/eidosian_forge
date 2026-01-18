import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
def _get_attr(obj_, name_):
    if bypass_descriptor_protocol:
        return object.__getattribute__(obj_, name_)
    else:
        return getattr(obj_, name_)