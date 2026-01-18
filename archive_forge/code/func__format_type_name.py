import collections
import copy
import importlib.metadata
import json
import logging
import operator
import sys
import yaml
from oslo_config import cfg
from oslo_i18n import _message
import stevedore.named  # noqa
def _format_type_name(opt_type):
    """Format the type name to use in describing an option"""
    try:
        return opt_type.type_name
    except AttributeError:
        pass
    try:
        return _TYPE_NAMES[opt_type]
    except KeyError:
        pass
    return 'unknown value'