from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class HeaderRegistry:
    """A header_factory and header registry."""

    def __init__(self, base_class=BaseHeader, default_class=UnstructuredHeader, use_default_map=True):
        """Create a header_factory that works with the Policy API.

        base_class is the class that will be the last class in the created
        header class's __bases__ list.  default_class is the class that will be
        used if "name" (see __call__) does not appear in the registry.
        use_default_map controls whether or not the default mapping of names to
        specialized classes is copied in to the registry when the factory is
        created.  The default is True.

        """
        self.registry = {}
        self.base_class = base_class
        self.default_class = default_class
        if use_default_map:
            self.registry.update(_default_header_map)

    def map_to_type(self, name, cls):
        """Register cls as the specialized class for handling "name" headers.

        """
        self.registry[name.lower()] = cls

    def __getitem__(self, name):
        cls = self.registry.get(name.lower(), self.default_class)
        return type('_' + cls.__name__, (cls, self.base_class), {})

    def __call__(self, name, value):
        """Create a header instance for header 'name' from 'value'.

        Creates a header instance by creating a specialized class for parsing
        and representing the specified header by combining the factory
        base_class with a specialized class from the registry or the
        default_class, and passing the name and value to the constructed
        class's constructor.

        """
        return self[name](name, value)