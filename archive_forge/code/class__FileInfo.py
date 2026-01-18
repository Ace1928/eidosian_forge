import collections
import inspect
import logging
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import reflection
from proto.marshal.rules.message import MessageRule
class _FileInfo(collections.namedtuple('_FileInfo', ['descriptor', 'messages', 'enums', 'name', 'nested', 'nested_enum'])):
    registry = {}

    @classmethod
    def maybe_add_descriptor(cls, filename, package):
        descriptor = cls.registry.get(filename)
        if not descriptor:
            descriptor = cls.registry[filename] = cls(descriptor=descriptor_pb2.FileDescriptorProto(name=filename, package=package, syntax='proto3'), enums=collections.OrderedDict(), messages=collections.OrderedDict(), name=filename, nested={}, nested_enum={})
        return descriptor

    @staticmethod
    def proto_file_name(name):
        return '{0}.proto'.format(name.replace('.', '/'))

    def _get_manifest(self, new_class):
        module = inspect.getmodule(new_class)
        if hasattr(module, '__protobuf__'):
            return frozenset(module.__protobuf__.manifest)
        return frozenset()

    def _get_remaining_manifest(self, new_class):
        return self._get_manifest(new_class) - {new_class.__name__}

    def _calculate_salt(self, new_class, fallback):
        manifest = self._get_manifest(new_class)
        if manifest and new_class.__name__ not in manifest:
            log.warning('proto-plus module {module} has a declared manifest but {class_name} is not in it'.format(module=inspect.getmodule(new_class).__name__, class_name=new_class.__name__))
        return '' if new_class.__name__ in manifest else (fallback or '').lower()

    def generate_file_pb(self, new_class, fallback_salt=''):
        """Generate the descriptors for all protos in the file.

        This method takes the file descriptor attached to the parent
        message and generates the immutable descriptors for all of the
        messages in the file descriptor. (This must be done in one fell
        swoop for immutability and to resolve proto cross-referencing.)

        This is run automatically when the last proto in the file is
        generated, as determined by the module's __all__ tuple.
        """
        pool = descriptor_pool.Default()
        salt = self._calculate_salt(new_class, fallback_salt)
        self.descriptor.name = '{name}.proto'.format(name='_'.join([self.descriptor.name[:-6], salt]).rstrip('_'))
        pool.Add(self.descriptor)
        for full_name, proto_plus_message in self.messages.items():
            descriptor = pool.FindMessageTypeByName(full_name)
            pb_message = reflection.GeneratedProtocolMessageType(descriptor.name, (message.Message,), {'DESCRIPTOR': descriptor, '__module__': None})
            proto_plus_message._meta._pb = pb_message
            proto_plus_message._meta.marshal.register(pb_message, MessageRule(pb_message, proto_plus_message))
            for field in proto_plus_message._meta.fields.values():
                if field.message and isinstance(field.message, str):
                    field.message = self.messages[field.message]
                elif field.enum and isinstance(field.enum, str):
                    field.enum = self.enums[field.enum]
        for full_name, proto_plus_enum in self.enums.items():
            descriptor = pool.FindEnumTypeByName(full_name)
            proto_plus_enum._meta.pb = descriptor
        self.registry.pop(self.name)

    def ready(self, new_class):
        """Return True if a file descriptor may added, False otherwise.

        This determine if all the messages that we plan to create have been
        created, as best as we are able.

        Since messages depend on one another, we create descriptor protos
        (which reference each other using strings) and wait until we have
        built everything that is going to be in the module, and then
        use the descriptor protos to instantiate the actual descriptors in
        one fell swoop.

        Args:
            new_class (~.MessageMeta): The new class currently undergoing
                creation.
        """
        if len(self.nested) or len(self.nested_enum):
            return False
        for field in self.unresolved_fields:
            if field.message and field.message not in self.messages or (field.enum and field.enum not in self.enums):
                return False
        module = inspect.getmodule(new_class)
        manifest = self._get_remaining_manifest(new_class)
        return all((hasattr(module, i) for i in manifest))

    @property
    def unresolved_fields(self):
        """Return fields with referencing message types as strings."""
        for proto_plus_message in self.messages.values():
            for field in proto_plus_message._meta.fields.values():
                if field.message and isinstance(field.message, str) or (field.enum and isinstance(field.enum, str)):
                    yield field