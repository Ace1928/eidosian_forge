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
def _format_defaults(opt):
    """Return a list of formatted default values."""
    if isinstance(opt, cfg.MultiStrOpt):
        if opt.sample_default is not None:
            defaults = opt.sample_default
        elif not opt.default:
            defaults = ['']
        else:
            defaults = opt.default
    else:
        if opt.sample_default is not None:
            default_str = str(opt.sample_default)
        elif opt.default is None:
            default_str = '<None>'
        elif isinstance(opt, (cfg.StrOpt, cfg.IntOpt, cfg.FloatOpt, cfg.IPOpt, cfg.PortOpt, cfg.HostnameOpt, cfg.HostAddressOpt, cfg.URIOpt, cfg.Opt)):
            default_str = str(opt.default)
        elif isinstance(opt, cfg.BoolOpt):
            default_str = str(opt.default).lower()
        elif isinstance(opt, (cfg.ListOpt, cfg._ConfigFileOpt, cfg._ConfigDirOpt)):
            default_str = ','.join((str(d) for d in opt.default))
        elif isinstance(opt, cfg.DictOpt):
            sorted_items = sorted(opt.default.items(), key=operator.itemgetter(0))
            default_str = ','.join(['%s:%s' % i for i in sorted_items])
        else:
            LOG.warning('Unknown option type: %s', repr(opt))
            default_str = str(opt.default)
        defaults = [default_str]
    results = []
    for default_str in defaults:
        if not isinstance(default_str, str):
            default_str = str(default_str)
        if default_str.strip() != default_str:
            default_str = '"%s"' % default_str
        results.append(default_str)
    return results