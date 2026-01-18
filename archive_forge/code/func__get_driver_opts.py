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
def _get_driver_opts(driver_option_name, namespaces):
    """List the options available from plugins for drivers based on the option.

    :param driver_option_name: The name of the option controlling the
                               driver options.
    :param namespaces: a list of namespaces registered under
                       'oslo.config.opts.' + driver_option_name
    :returns: a dict mapping driver name to option list

    """
    all_opts = {}
    loaders = _get_driver_opts_loaders(namespaces, driver_option_name)
    for plugin_name, loader in loaders:
        for driver_name, option_list in loader().items():
            all_opts.setdefault(driver_name, []).extend(option_list)
    return all_opts