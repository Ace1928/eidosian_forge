import collections
import copy
import itertools
import random
import re
import warnings
def _object_matcher(obj):
    """Retrieve a matcher function by passing an arbitrary object (PRIVATE).

    Passing a ``TreeElement`` such as a ``Clade`` or ``Tree`` instance returns
    an identity matcher, passing a type such as the ``PhyloXML.Taxonomy`` class
    returns a class matcher, and passing a dictionary returns an attribute
    matcher.

    The resulting 'match' function returns True when given an object matching
    the specification (identity, type or attribute values), otherwise False.
    This is useful for writing functions that search the tree, and probably
    shouldn't be used directly by the end user.
    """
    if isinstance(obj, TreeElement):
        return _identity_matcher(obj)
    if isinstance(obj, type):
        return _class_matcher(obj)
    if isinstance(obj, str):
        return _string_matcher(obj)
    if isinstance(obj, dict):
        return _attribute_matcher(obj)
    if callable(obj):
        return _function_matcher(obj)
    raise ValueError(f'{obj} (type {type(obj)}) is not a valid type for comparison.')