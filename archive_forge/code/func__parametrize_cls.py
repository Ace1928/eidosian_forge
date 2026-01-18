from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def _parametrize_cls(module, cls):
    """implement a class-based version of pytest parametrize."""
    if '_sa_parametrize' not in cls.__dict__:
        return [cls]
    _sa_parametrize = cls._sa_parametrize
    classes = []
    for full_param_set in itertools.product(*[params for argname, params in _sa_parametrize]):
        cls_variables = {}
        for argname, param in zip([_sa_param[0] for _sa_param in _sa_parametrize], full_param_set):
            if not argname:
                raise TypeError('need argnames for class-based combinations')
            argname_split = re.split(',\\s*', argname)
            for arg, val in zip(argname_split, param.values):
                cls_variables[arg] = val
        parametrized_name = '_'.join((re.sub('\\W', '', token) for param in full_param_set for token in param.id.split('-')))
        name = '%s_%s' % (cls.__name__, parametrized_name)
        newcls = type.__new__(type, name, (cls,), cls_variables)
        setattr(module, name, newcls)
        classes.append(newcls)
    return classes