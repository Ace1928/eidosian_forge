import collections
import itertools
import sys
from oslo_config import cfg
from oslo_log import log
from heat.common import plugin_loader
class PluginMapping(object):
    """A class for managing plugin mappings."""

    def __init__(self, names, *args, **kwargs):
        """Initialise with the mapping name(s) and arguments.

        `names` can be a single name or a list of names. The first name found
        in a given module is the one used. Each module is searched for a
        function called <name>_mapping() which is called to retrieve the
        mappings provided by that module. Any other arguments passed will be
        passed to the mapping functions.
        """
        if isinstance(names, str):
            names = [names]
        self.names = ['%s_mapping' % name for name in names]
        self.args = args
        self.kwargs = kwargs

    def load_from_module(self, module):
        """Return the mapping specified in the given module.

        If no such mapping is specified, an empty dictionary is returned.
        """
        for mapping_name in self.names:
            mapping_func = getattr(module, mapping_name, None)
            if callable(mapping_func):
                fmt_data = {'mapping_name': mapping_name, 'module': module}
                try:
                    mapping_dict = mapping_func(*self.args, **self.kwargs)
                except Exception:
                    LOG.error('Failed to load %(mapping_name)s from %(module)s', fmt_data)
                    raise
                else:
                    if isinstance(mapping_dict, collections.abc.Mapping):
                        return mapping_dict
                    elif mapping_dict is not None:
                        LOG.error('Invalid type for %(mapping_name)s from %(module)s', fmt_data)
        return {}

    def load_all(self, plugin_manager):
        """Iterate over the mappings from all modules in the plugin manager.

        Mappings are returned as a list of (key, value) tuples.
        """
        mod_dicts = plugin_manager.map_to_modules(self.load_from_module)
        return itertools.chain.from_iterable((d.items() for d in mod_dicts))