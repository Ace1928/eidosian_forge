import os
import warnings
import builtins
import cherrypy
def _known_ns(self, app):
    ns = ['wsgi']
    ns.extend(app.toolboxes)
    ns.extend(app.namespaces)
    ns.extend(app.request_class.namespaces)
    ns.extend(cherrypy.config.namespaces)
    ns += self.extra_config_namespaces
    for section, conf in app.config.items():
        is_path_section = section.startswith('/')
        if is_path_section and isinstance(conf, dict):
            for k in conf:
                atoms = k.split('.')
                if len(atoms) > 1:
                    if atoms[0] not in ns:
                        if atoms[0] == 'cherrypy' and atoms[1] in ns:
                            msg = 'The config entry %r is invalid; try %r instead.\nsection: [%s]' % (k, '.'.join(atoms[1:]), section)
                        else:
                            msg = 'The config entry %r is invalid, because the %r config namespace is unknown.\nsection: [%s]' % (k, atoms[0], section)
                        warnings.warn(msg)
                    elif atoms[0] == 'tools':
                        if atoms[1] not in dir(cherrypy.tools):
                            msg = 'The config entry %r may be invalid, because the %r tool was not found.\nsection: [%s]' % (k, atoms[1], section)
                            warnings.warn(msg)