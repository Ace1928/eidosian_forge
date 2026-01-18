from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
class UnconfiguredEdits(AbstractEdits):
    """Maps key to edit functions, and bins them by what parameters they take.

    Only functions with specific signatures can be added:
        * func(**kwargs) -> cursor_offset, line
        * func(**kwargs) -> cursor_offset, line, cut_buffer
        where kwargs are in among the keys of Edits.default_kwargs
    These functions will be run to determine their return type, so no side
    effects!

    More concrete Edits instances can be created by applying a config with
    Edits.mapping_with_config() - this creates a new Edits instance
    that uses a config file to assign config_attr bindings.

    Keys can't be added twice, config attributes can't be added twice.
    """

    def mapping_with_config(self, config, key_dispatch):
        """Creates a new mapping object by applying a config object"""
        return ConfiguredEdits(self.simple_edits, self.cut_buffer_edits, self.awaiting_config, config, key_dispatch)

    def on(self, key=None, config=None):
        if not (key is None) ^ (config is None):
            raise ValueError('Must use exactly one of key, config')
        if key is not None:

            def add_to_keybinds(func):
                self.add(key, func)
                return func
            return add_to_keybinds
        else:

            def add_to_config(func):
                self.add_config_attr(config, func)
                return func
            return add_to_config