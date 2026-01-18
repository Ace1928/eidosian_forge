import os
import warnings
import builtins
import cherrypy
def _populate_known_types(self):
    b = [x for x in vars(builtins).values() if type(x) is type(str)]

    def traverse(obj, namespace):
        for name in dir(obj):
            if name == 'body_params':
                continue
            vtype = type(getattr(obj, name, None))
            if vtype in b:
                self.known_config_types[namespace + '.' + name] = vtype
    traverse(cherrypy.request, 'request')
    traverse(cherrypy.response, 'response')
    traverse(cherrypy.server, 'server')
    traverse(cherrypy.engine, 'engine')
    traverse(cherrypy.log, 'log')