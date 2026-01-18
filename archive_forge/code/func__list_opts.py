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
def _list_opts(namespaces):
    """List the options available via the given namespaces.

    Duplicate options from a namespace are removed.

    :param namespaces: a list of namespaces registered under 'oslo.config.opts'
    :returns: a list of (namespace, [(group, [opt_1, opt_2])]) tuples
    """
    loaders = _get_raw_opts_loaders(namespaces)
    _update_defaults(namespaces)
    response = []
    for namespace, loader in loaders:
        namespace_values = []
        for group, group_opts in loader():
            group_opts = list(group_opts)
            if isinstance(group, cfg.OptGroup):
                if group.driver_option:
                    driver_opts = _get_driver_opts(group.driver_option, namespaces)
                    driver_opt_names = {}
                    for driver_name, opts in sorted(driver_opts.items()):
                        driver_opt_names.setdefault(driver_name, []).extend((o.name for o in opts))
                        group_opts.extend(opts)
                    group._save_driver_opts(driver_opt_names)
            namespace_values.append((group, group_opts))
        response.append((namespace, namespace_values))
    return _cleanup_opts(response)