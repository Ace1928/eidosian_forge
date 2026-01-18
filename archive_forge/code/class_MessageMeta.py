import collections
import collections.abc
import copy
import re
from typing import List, Type
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf.json_format import MessageToDict, MessageToJson, Parse
from proto import _file_info
from proto import _package_info
from proto.fields import Field
from proto.fields import MapField
from proto.fields import RepeatedField
from proto.marshal import Marshal
from proto.primitives import ProtoType
from proto.utils import has_upb
class MessageMeta(type):
    """A metaclass for building and registering Message subclasses."""

    def __new__(mcls, name, bases, attrs):
        if not bases:
            return super().__new__(mcls, name, bases, attrs)
        package, marshal = _package_info.compile(name, attrs)
        local_path = tuple(attrs.get('__qualname__', name).split('.'))
        if '<locals>' in local_path:
            ix = local_path.index('<locals>')
            local_path = local_path[:ix - 1] + local_path[ix + 1:]
        full_name = '.'.join((package,) + local_path).lstrip('.')
        map_fields = {}
        for key, field in attrs.items():
            if not isinstance(field, MapField):
                continue
            msg_name = '{pascal_key}Entry'.format(pascal_key=re.sub('_\\w', lambda m: m.group()[1:].upper(), key).replace(key[0], key[0].upper(), 1))
            entry_attrs = collections.OrderedDict({'__module__': attrs.get('__module__', None), '__qualname__': '{prefix}.{name}'.format(prefix=attrs.get('__qualname__', name), name=msg_name), '_pb_options': {'map_entry': True}})
            entry_attrs['key'] = Field(field.map_key_type, number=1)
            entry_attrs['value'] = Field(field.proto_type, number=2, enum=field.enum, message=field.message)
            map_fields[msg_name] = MessageMeta(msg_name, (Message,), entry_attrs)
            map_fields[key] = RepeatedField(ProtoType.MESSAGE, number=field.number, message=map_fields[msg_name])
        attrs.update(map_fields)
        fields = []
        new_attrs = {}
        oneofs = collections.OrderedDict()
        proto_imports = set()
        index = 0
        for key, field in attrs.items():
            if not isinstance(field, Field):
                new_attrs[key] = field
                continue
            field.mcls_data = {'name': key, 'parent_name': full_name, 'index': index, 'package': package}
            fields.append(field)
            if field.oneof:
                oneofs.setdefault(field.oneof, len(oneofs))
                field.descriptor.oneof_index = oneofs[field.oneof]
            if field.message and (not isinstance(field.message, str)):
                field_msg = field.message
                if hasattr(field_msg, 'pb') and callable(field_msg.pb):
                    field_msg = field_msg.pb()
                if field_msg:
                    proto_imports.add(field_msg.DESCRIPTOR.file.name)
            elif field.enum and (not isinstance(field.enum, str)):
                field_enum = field.enum._meta.pb if hasattr(field.enum, '_meta') else field.enum.DESCRIPTOR
                if field_enum:
                    proto_imports.add(field_enum.file.name)
            index += 1
        opt_attrs = {}
        for field in fields:
            if field.optional:
                field.oneof = '_{}'.format(field.name)
                field.descriptor.oneof_index = oneofs[field.oneof] = len(oneofs)
                opt_attrs[field.name] = field.name
        if opt_attrs:
            mcls = type('AttrsMeta', (mcls,), opt_attrs)
        filename = _file_info._FileInfo.proto_file_name(new_attrs.get('__module__', name.lower()))
        file_info = _file_info._FileInfo.maybe_add_descriptor(filename, package)
        for proto_import in proto_imports:
            if proto_import not in file_info.descriptor.dependency:
                file_info.descriptor.dependency.append(proto_import)
        opts = descriptor_pb2.MessageOptions(**new_attrs.pop('_pb_options', {}))
        desc = descriptor_pb2.DescriptorProto(name=name, field=[i.descriptor for i in fields], oneof_decl=[descriptor_pb2.OneofDescriptorProto(name=i) for i in oneofs.keys()], options=opts)
        child_paths = [p for p in file_info.nested.keys() if local_path == p[:-1]]
        for child_path in child_paths:
            desc.nested_type.add().MergeFrom(file_info.nested.pop(child_path))
        child_paths = [p for p in file_info.nested_enum.keys() if local_path == p[:-1]]
        for child_path in child_paths:
            desc.enum_type.add().MergeFrom(file_info.nested_enum.pop(child_path))
        if len(local_path) == 1:
            file_info.descriptor.message_type.add().MergeFrom(desc)
        else:
            file_info.nested[local_path] = desc
        new_attrs['_meta'] = _MessageInfo(fields=fields, full_name=full_name, marshal=marshal, options=opts, package=package)
        cls = super().__new__(mcls, name, bases, new_attrs)
        cls._meta.parent = cls
        for field in cls._meta.fields.values():
            field.parent = cls
        file_info.messages[full_name] = cls
        if file_info.ready(new_class=cls):
            file_info.generate_file_pb(new_class=cls, fallback_salt=full_name)
        return cls

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs):
        return collections.OrderedDict()

    @property
    def meta(cls):
        return cls._meta

    def __dir__(self):
        try:
            names = set(dir(type))
            names.update(('meta', 'pb', 'wrap', 'serialize', 'deserialize', 'to_json', 'from_json', 'to_dict', 'copy_from'))
            desc = self.pb().DESCRIPTOR
            names.update((t.name for t in desc.nested_types))
            names.update((e.name for e in desc.enum_types))
            return names
        except AttributeError:
            return dir(type)

    def pb(cls, obj=None, *, coerce: bool=False):
        """Return the underlying protobuf Message class or instance.

        Args:
            obj: If provided, and an instance of ``cls``, return the
                underlying protobuf instance.
            coerce (bool): If provided, will attempt to coerce ``obj`` to
                ``cls`` if it is not already an instance.
        """
        if obj is None:
            return cls.meta.pb
        if not isinstance(obj, cls):
            if coerce:
                obj = cls(obj)
            else:
                raise TypeError('%r is not an instance of %s' % (obj, cls.__name__))
        return obj._pb

    def wrap(cls, pb):
        """Return a Message object that shallowly wraps the descriptor.

        Args:
            pb: A protocol buffer object, such as would be returned by
                :meth:`pb`.
        """
        instance = cls.__new__(cls)
        super(cls, instance).__setattr__('_pb', pb)
        return instance

    def serialize(cls, instance) -> bytes:
        """Return the serialized proto.

        Args:
            instance: An instance of this message type, or something
                compatible (accepted by the type's constructor).

        Returns:
            bytes: The serialized representation of the protocol buffer.
        """
        return cls.pb(instance, coerce=True).SerializeToString()

    def deserialize(cls, payload: bytes) -> 'Message':
        """Given a serialized proto, deserialize it into a Message instance.

        Args:
            payload (bytes): The serialized proto.

        Returns:
            ~.Message: An instance of the message class against which this
            method was called.
        """
        return cls.wrap(cls.pb().FromString(payload))

    def to_json(cls, instance, *, use_integers_for_enums=True, including_default_value_fields=True, preserving_proto_field_name=False, indent=2) -> str:
        """Given a message instance, serialize it to json

        Args:
            instance: An instance of this message type, or something
                compatible (accepted by the type's constructor).
            use_integers_for_enums (Optional(bool)): An option that determines whether enum
                values should be represented by strings (False) or integers (True).
                Default is True.
            preserving_proto_field_name (Optional(bool)): An option that
                determines whether field name representations preserve
                proto case (snake_case) or use lowerCamelCase. Default is False.
            indent: The JSON object will be pretty-printed with this indent level.
                An indent level of 0 or negative will only insert newlines.
                Pass None for the most compact representation without newlines.

        Returns:
            str: The json string representation of the protocol buffer.
        """
        return MessageToJson(cls.pb(instance), use_integers_for_enums=use_integers_for_enums, including_default_value_fields=including_default_value_fields, preserving_proto_field_name=preserving_proto_field_name, indent=indent)

    def from_json(cls, payload, *, ignore_unknown_fields=False) -> 'Message':
        """Given a json string representing an instance,
        parse it into a message.

        Args:
            paylod: A json string representing a message.
            ignore_unknown_fields (Optional(bool)): If True, do not raise errors
                for unknown fields.

        Returns:
            ~.Message: An instance of the message class against which this
            method was called.
        """
        instance = cls()
        Parse(payload, instance._pb, ignore_unknown_fields=ignore_unknown_fields)
        return instance

    def to_dict(cls, instance, *, use_integers_for_enums=True, preserving_proto_field_name=True, including_default_value_fields=True) -> 'Message':
        """Given a message instance, return its representation as a python dict.

        Args:
            instance: An instance of this message type, or something
                      compatible (accepted by the type's constructor).
            use_integers_for_enums (Optional(bool)): An option that determines whether enum
                values should be represented by strings (False) or integers (True).
                Default is True.
            preserving_proto_field_name (Optional(bool)): An option that
                determines whether field name representations preserve
                proto case (snake_case) or use lowerCamelCase. Default is True.
            including_default_value_fields (Optional(bool)): An option that
                determines whether the default field values should be included in the results.
                Default is True.

        Returns:
            dict: A representation of the protocol buffer using pythonic data structures.
                  Messages and map fields are represented as dicts,
                  repeated fields are represented as lists.
        """
        return MessageToDict(cls.pb(instance), including_default_value_fields=including_default_value_fields, preserving_proto_field_name=preserving_proto_field_name, use_integers_for_enums=use_integers_for_enums)

    def copy_from(cls, instance, other):
        """Equivalent for protobuf.Message.CopyFrom

        Args:
            instance: An instance of this message type
            other: (Union[dict, ~.Message):
                A dictionary or message to reinitialize the values for this message.
        """
        if isinstance(other, cls):
            other = Message.pb(other)
        elif isinstance(other, cls.pb()):
            pass
        elif isinstance(other, collections.abc.Mapping):
            other = cls._meta.pb(**other)
        else:
            raise TypeError('invalid argument type to copy to {}: {}'.format(cls.__name__, other.__class__.__name__))
        cls.pb(instance).CopyFrom(other)