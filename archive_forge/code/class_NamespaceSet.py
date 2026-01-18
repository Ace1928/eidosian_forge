import builtins
import configparser
import operator
import sys
from cherrypy._cpcompat import text_or_bytes
class NamespaceSet(dict):
    """A dict of config namespace names and handlers.

    Each config entry should begin with a namespace name; the corresponding
    namespace handler will be called once for each config entry in that
    namespace, and will be passed two arguments: the config key (with the
    namespace removed) and the config value.

    Namespace handlers may be any Python callable; they may also be
    context managers, in which case their __enter__
    method should return a callable to be used as the handler.
    See cherrypy.tools (the Toolbox class) for an example.
    """

    def __call__(self, config):
        """Iterate through config and pass it to each namespace handler.

        config
            A flat dict, where keys use dots to separate
            namespaces, and values are arbitrary.

        The first name in each config key is used to look up the corresponding
        namespace handler. For example, a config entry of {'tools.gzip.on': v}
        will call the 'tools' namespace handler with the args: ('gzip.on', v)
        """
        ns_confs = {}
        for k in config:
            if '.' in k:
                ns, name = k.split('.', 1)
                bucket = ns_confs.setdefault(ns, {})
                bucket[name] = config[k]
        for ns, handler in self.items():
            exit = getattr(handler, '__exit__', None)
            if exit:
                callable = handler.__enter__()
                no_exc = True
                try:
                    try:
                        for k, v in ns_confs.get(ns, {}).items():
                            callable(k, v)
                    except Exception:
                        no_exc = False
                        if exit is None:
                            raise
                        if not exit(*sys.exc_info()):
                            raise
                finally:
                    if no_exc and exit:
                        exit(None, None, None)
            else:
                for k, v in ns_confs.get(ns, {}).items():
                    handler(k, v)

    def __repr__(self):
        return '%s.%s(%s)' % (self.__module__, self.__class__.__name__, dict.__repr__(self))

    def __copy__(self):
        newobj = self.__class__()
        newobj.update(self)
        return newobj
    copy = __copy__