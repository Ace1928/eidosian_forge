import pkg_resources
from oslo_config import cfg
from oslo_log import log as logging
import pbr
from keystonemiddleware import exceptions
from keystonemiddleware.i18n import _
def _conf_values_type_convert(group_name, all_options, conf):
    """Convert conf values into correct type."""
    if not conf:
        return {}
    opts = {}
    opt_types = {}
    for group, options in all_options:
        if group != group_name:
            continue
        for o in options:
            type_dest = (getattr(o, 'type', str), o.dest)
            opt_types[o.dest] = type_dest
            for d_o in o.deprecated_opts:
                opt_types[d_o.name] = type_dest
        break
    for k, v in conf.items():
        dest = k
        try:
            if v is not None and k not in ['here', '__file__', 'configkey']:
                type_, dest = opt_types[k]
                v = type_(v)
        except KeyError:
            _LOG.warning('The option "%s" is not known to keystonemiddleware', k)
        except ValueError as e:
            raise exceptions.ConfigurationError(_('Unable to convert the value of option "%(key)s" into correct type: %(ex)s') % {'key': k, 'ex': e})
        opts[dest] = v
    return opts