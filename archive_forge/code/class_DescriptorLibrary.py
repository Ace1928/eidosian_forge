import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
class DescriptorLibrary(object):
    """A descriptor library is an object that contains known definitions.

    A descriptor library contains a cache of descriptor objects mapped by
    definition name.  It contains all types of descriptors except for
    file sets.

    When a definition name is requested that the library does not know about
    it can be provided with a descriptor loader which attempt to resolve the
    missing descriptor.
    """

    @util.positional(1)
    def __init__(self, descriptors=None, descriptor_loader=import_descriptor_loader):
        """Constructor.

        Args:
          descriptors: A dictionary or dictionary-like object that can be used
            to store and cache descriptors by definition name.
          definition_loader: A function used for resolving missing descriptors.
            The function takes a definition name as its parameter and returns
            an appropriate descriptor.  It may raise DefinitionNotFoundError.
        """
        self.__descriptor_loader = descriptor_loader
        self.__descriptors = descriptors or {}

    def lookup_descriptor(self, definition_name):
        """Lookup descriptor by name.

        Get descriptor from library by name.  If descriptor is not found will
        attempt to find via descriptor loader if provided.

        Args:
          definition_name: Definition name to find.

        Returns:
          Descriptor that describes definition name.

        Raises:
          DefinitionNotFoundError if not descriptor exists for definition name.
        """
        try:
            return self.__descriptors[definition_name]
        except KeyError:
            pass
        if self.__descriptor_loader:
            definition = self.__descriptor_loader(definition_name)
            self.__descriptors[definition_name] = definition
            return definition
        else:
            raise messages.DefinitionNotFoundError('Could not find definition for %s' % definition_name)

    def lookup_package(self, definition_name):
        """Determines the package name for any definition.

        Determine the package that any definition name belongs to. May
        check parent for package name and will resolve missing
        descriptors if provided descriptor loader.

        Args:
          definition_name: Definition name to find package for.

        """
        while True:
            descriptor = self.lookup_descriptor(definition_name)
            if isinstance(descriptor, FileDescriptor):
                return descriptor.package
            else:
                index = definition_name.rfind('.')
                if index < 0:
                    return None
                definition_name = definition_name[:index]