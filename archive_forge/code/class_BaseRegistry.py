from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.filters import CLIFilter, to_cli_filter, Never
from prompt_toolkit.keys import Key, Keys
from six import text_type, with_metaclass
class BaseRegistry(with_metaclass(ABCMeta, object)):
    """
    Interface for a Registry.
    """
    _version = 0

    @abstractmethod
    def get_bindings_for_keys(self, keys):
        pass

    @abstractmethod
    def get_bindings_starting_with_keys(self, keys):
        pass