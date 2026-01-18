from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
def _remove_dups_flatten(parameters):
    """Internal helper for Union creation and substitution.

    Flatten Unions among parameters, then remove duplicates.
    """
    params = []
    for p in parameters:
        if isinstance(p, (_UnionGenericAlias, types.UnionType)):
            params.extend(p.__args__)
        else:
            params.append(p)
    return tuple(_deduplicate(params))