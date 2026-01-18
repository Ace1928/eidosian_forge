from __future__ import absolute_import, division, print_function
import os
class HashiVaultOptionAdapter(object):
    """
    The purpose of this class is to provide a standard interface for option getting/setting
    within module_utils code, since the processes are so different in plugins and modules.

    Attention is paid to ensuring that in plugins we use the methods provided by Config Manager,
    but to allow flexibility to create an adapter to work with other sources, hence the design
    of defining specific methods exposed, and having them call provided function references.
    """

    @classmethod
    def from_dict(cls, dict):
        return cls(getter=dict.__getitem__, setter=dict.__setitem__, haver=lambda key: key in dict, updater=dict.update, defaultsetter=dict.setdefault, defaultgetter=dict.get)

    @classmethod
    def from_ansible_plugin(cls, plugin):
        return cls(getter=plugin.get_option, setter=plugin.set_option, haver=plugin.has_option if hasattr(plugin, 'has_option') else None)

    @classmethod
    def from_ansible_module(cls, module):
        return cls.from_dict(module.params)

    def __init__(self, getter, setter, haver=None, updater=None, getitems=None, getfiltereditems=None, getfilleditems=None, defaultsetter=None, defaultgetter=None):

        def _default_default_setter(key, default=None):
            try:
                value = self.get_option(key)
                return value
            except KeyError:
                self.set_option(key, default)
                return default

        def _default_updater(**kwargs):
            for key, value in kwargs.items():
                self.set_option(key, value)

        def _default_haver(key):
            try:
                self.get_option(key)
                return True
            except KeyError:
                return False

        def _default_getitems(*args):
            return dict(((key, self.get_option(key)) for key in args))

        def _default_getfiltereditems(filter, *args):
            return dict(((key, value) for key, value in self.get_options(*args).items() if filter(key, value)))

        def _default_getfilleditems(*args):
            return self.get_filtered_options(lambda k, v: v is not None, *args)

        def _default_default_getter(key, default):
            try:
                return self.get_option(key)
            except KeyError:
                return default
        self._getter = getter
        self._setter = setter
        self._haver = haver or _default_haver
        self._updater = updater or _default_updater
        self._getitems = getitems or _default_getitems
        self._getfiltereditems = getfiltereditems or _default_getfiltereditems
        self._getfilleditems = getfilleditems or _default_getfilleditems
        self._defaultsetter = defaultsetter or _default_default_setter
        self._defaultgetter = defaultgetter or _default_default_getter

    def get_option(self, key):
        return self._getter(key)

    def get_option_default(self, key, default=None):
        return self._defaultgetter(key, default)

    def set_option(self, key, value):
        return self._setter(key, value)

    def set_option_default(self, key, default=None):
        return self._defaultsetter(key, default)

    def has_option(self, key):
        return self._haver(key)

    def set_options(self, **kwargs):
        return self._updater(**kwargs)

    def get_options(self, *args):
        return self._getitems(*args)

    def get_filtered_options(self, filter, *args):
        return self._getfiltereditems(filter, *args)

    def get_filled_options(self, *args):
        return self._getfilleditems(*args)