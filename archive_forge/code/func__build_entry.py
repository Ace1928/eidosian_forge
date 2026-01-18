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
def _build_entry(opt, group, namespace, conf):
    """Return a dict representing the passed in opt

    The dict will contain all public attributes of opt, as well as additional
    entries for namespace, choices, min, and max.  Any DeprecatedOpts
    contained in the deprecated_opts member will be converted to a dict with
    the format: {'group': <deprecated group>, 'name': <deprecated name>}

    :param opt: The Opt object to represent as a dict.
    :param group: The name of the group containing opt.
    :param namespace: The name of the namespace containing opt.
    :param conf: The ConfigOpts object containing the options for the
                 generator tool
    """
    entry = {key: value for key, value in opt.__dict__.items() if not key.startswith('_')}
    entry['namespace'] = namespace
    if getattr(entry['type'], 'choices', None):
        entry['choices'] = list(entry['type'].choices.items())
    else:
        entry['choices'] = []
    entry['min'] = getattr(entry['type'], 'min', None)
    entry['max'] = getattr(entry['type'], 'max', None)
    entry['type'] = _format_type_name(entry['type'])
    deprecated_opts = []
    for deprecated_opt in entry['deprecated_opts']:
        if not deprecated_opt.name or '-' not in deprecated_opt.name:
            deprecated_opts.append({'group': deprecated_opt.group or group, 'name': deprecated_opt.name or entry['name']})
    entry['deprecated_opts'] = deprecated_opts
    return entry