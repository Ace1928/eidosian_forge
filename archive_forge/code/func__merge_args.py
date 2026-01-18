import abc
import argparse
import functools
import logging
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def _merge_args(qCmd, parsed_args, _extra_values, value_specs):
    """Merge arguments from _extra_values into parsed_args.

    If an argument value are provided in both and it is a list,
    the values in _extra_values will be merged into parsed_args.

    @param parsed_args: the parsed args from known options
    @param _extra_values: the other parsed arguments in unknown parts
    @param values_specs: the unparsed unknown parts
    """
    temp_values = _extra_values.copy()
    for key, value in temp_values.items():
        if hasattr(parsed_args, key):
            arg_value = getattr(parsed_args, key)
            if arg_value is not None and value is not None:
                if isinstance(arg_value, list):
                    if value and isinstance(value, list):
                        if not arg_value or isinstance(arg_value[0], type(value[0])):
                            arg_value.extend(value)
                            _extra_values.pop(key)