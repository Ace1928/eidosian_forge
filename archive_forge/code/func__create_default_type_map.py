from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_extension_utils
from _pydevd_bundle import pydevd_resolver
import sys
from _pydevd_bundle.pydevd_constants import BUILTINS_MODULE_NAME, MAXIMUM_VARIABLE_REPRESENTATION_SIZE, \
from _pydev_bundle.pydev_imports import quote
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_utils import isinstance_checked, hasattr_checked, DAPGrouper
from _pydevd_bundle.pydevd_resolver import get_var_scope, MoreItems, MoreItemsRange
from typing import Optional
def _create_default_type_map():
    default_type_map = [(type(None), None), (int, None), (float, None), (complex, None), (str, None), (tuple, pydevd_resolver.tupleResolver), (list, pydevd_resolver.tupleResolver), (dict, pydevd_resolver.dictResolver)]
    try:
        from collections import OrderedDict
        default_type_map.insert(0, (OrderedDict, pydevd_resolver.orderedDictResolver))
    except:
        pass
    try:
        default_type_map.append((long, None))
    except:
        pass
    default_type_map.append((DAPGrouper, pydevd_resolver.dapGrouperResolver))
    default_type_map.append((MoreItems, pydevd_resolver.forwardInternalResolverToObject))
    default_type_map.append((MoreItemsRange, pydevd_resolver.forwardInternalResolverToObject))
    try:
        default_type_map.append((set, pydevd_resolver.setResolver))
    except:
        pass
    try:
        default_type_map.append((frozenset, pydevd_resolver.setResolver))
    except:
        pass
    try:
        from django.utils.datastructures import MultiValueDict
        default_type_map.insert(0, (MultiValueDict, pydevd_resolver.multiValueDictResolver))
    except:
        pass
    try:
        from django.forms import BaseForm
        default_type_map.insert(0, (BaseForm, pydevd_resolver.djangoFormResolver))
    except:
        pass
    try:
        from collections import deque
        default_type_map.append((deque, pydevd_resolver.dequeResolver))
    except:
        pass
    try:
        from ctypes import Array
        default_type_map.append((Array, pydevd_resolver.tupleResolver))
    except:
        pass
    if frame_type is not None:
        default_type_map.append((frame_type, pydevd_resolver.frameResolver))
    if _IS_JYTHON:
        from org.python import core
        default_type_map.append((core.PyNone, None))
        default_type_map.append((core.PyInteger, None))
        default_type_map.append((core.PyLong, None))
        default_type_map.append((core.PyFloat, None))
        default_type_map.append((core.PyComplex, None))
        default_type_map.append((core.PyString, None))
        default_type_map.append((core.PyTuple, pydevd_resolver.tupleResolver))
        default_type_map.append((core.PyList, pydevd_resolver.tupleResolver))
        default_type_map.append((core.PyDictionary, pydevd_resolver.dictResolver))
        default_type_map.append((core.PyStringMap, pydevd_resolver.dictResolver))
        if hasattr(core, 'PyJavaInstance'):
            default_type_map.append((core.PyJavaInstance, pydevd_resolver.instanceResolver))
    return default_type_map