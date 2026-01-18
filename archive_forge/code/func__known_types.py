import os
import warnings
import builtins
import cherrypy
def _known_types(self, config):
    msg = 'The config entry %r in section %r is of type %r, which does not match the expected type %r.'
    for section, conf in config.items():
        if not isinstance(conf, dict):
            conf = {section: conf}
        for k, v in conf.items():
            if v is not None:
                expected_type = self.known_config_types.get(k, None)
                vtype = type(v)
                if expected_type and vtype != expected_type:
                    warnings.warn(msg % (k, section, vtype.__name__, expected_type.__name__))