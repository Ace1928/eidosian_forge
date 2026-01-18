from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import CLIFilter, to_cli_filter, Never
from prompt_toolkit.keys import Key, Keys
from six import text_type, with_metaclass
class ConditionalRegistry(_AddRemoveMixin):
    """
    Wraps around a `Registry`. Disable/enable all the key bindings according to
    the given (additional) filter.::

        @Condition
        def setting_is_true(cli):
            return True  # or False

        registy = ConditionalRegistry(registry, setting_is_true)

    When new key bindings are added to this object. They are also
    enable/disabled according to the given `filter`.

    :param registries: List of `Registry` objects.
    :param filter: `CLIFilter` object.
    """

    def __init__(self, registry=None, filter=True):
        registry = registry or Registry()
        assert isinstance(registry, BaseRegistry)
        _AddRemoveMixin.__init__(self)
        self.registry = registry
        self.filter = to_cli_filter(filter)

    def _update_cache(self):
        """ If the original registry was changed. Update our copy version. """
        expected_version = (self.registry._version, self._extra_registry._version)
        if self._last_version != expected_version:
            registry2 = Registry()
            for reg in (self.registry, self._extra_registry):
                for b in reg.key_bindings:
                    registry2.key_bindings.append(_Binding(keys=b.keys, handler=b.handler, filter=self.filter & b.filter, eager=b.eager, save_before=b.save_before))
            self._registry2 = registry2
            self._last_version = expected_version