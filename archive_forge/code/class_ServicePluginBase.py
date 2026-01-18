import abc
class ServicePluginBase(WorkerBase, metaclass=abc.ABCMeta):
    """Define base interface for any Advanced Service plugin."""
    supported_extension_aliases = []

    @classmethod
    def __subclasshook__(cls, klass):
        """Checking plugin class.

        The __subclasshook__ method is a class method
        that will be called every time a class is tested
        using issubclass(klass, ServicePluginBase).
        In that case, it will check that every method
        marked with the abstractmethod decorator is
        provided by the plugin class.
        """
        if not cls.__abstractmethods__:
            return NotImplemented
        for method in cls.__abstractmethods__:
            if any((method in base.__dict__ for base in klass.__mro__)):
                continue
            return NotImplemented
        return True

    @abc.abstractmethod
    def get_plugin_type(self):
        """Return one of predefined service types.

        """
        pass

    @abc.abstractmethod
    def get_plugin_description(self):
        """Return string description of the plugin."""
        pass