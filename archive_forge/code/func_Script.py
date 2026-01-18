from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import inspect
from fire import inspectutils
import six
def Script(name, component, default_options=None, shell='bash'):
    if shell == 'fish':
        return _FishScript(name, _Commands(component), default_options)
    return _BashScript(name, _Commands(component), default_options)