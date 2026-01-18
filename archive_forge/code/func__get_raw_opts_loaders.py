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
def _get_raw_opts_loaders(namespaces):
    """List the options available via the given namespaces.

    :param namespaces: a list of namespaces registered under 'oslo.config.opts'
    :returns: a list of (namespace, [(group, [opt_1, opt_2])]) tuples
    """
    mgr = stevedore.named.NamedExtensionManager('oslo.config.opts', names=namespaces, on_load_failure_callback=on_load_failure_callback, invoke_on_load=False)
    return [(e.name, e.plugin) for e in mgr]