import abc
import enum
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import duration_pb2
from cloudsdk.google.protobuf import timestamp_pb2
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import struct_pb2
from cloudsdk.google.protobuf import wrappers_pb2
from proto.marshal import compat
from proto.marshal.collections import MapComposite
from proto.marshal.collections import Repeated
from proto.marshal.collections import RepeatedComposite
from proto.marshal.rules import bytes as pb_bytes
from proto.marshal.rules import stringy_numbers
from proto.marshal.rules import dates
from proto.marshal.rules import struct
from proto.marshal.rules import wrappers
from proto.marshal.rules import field_mask
from proto.primitives import ProtoType
class BaseMarshal:
    """The base class to translate between protobuf and Python classes.

    Protocol buffers defines many common types (e.g. Timestamp, Duration)
    which also exist in the Python standard library. The marshal essentially
    translates between these: it keeps a registry of common protocol buffers
    and their Python representations, and translates back and forth.

    The protocol buffer class is always the "key" in this relationship; when
    presenting a message, the declared field types are used to determine
    whether a value should be transformed into another class. Similarly,
    when accepting a Python value (when setting a field, for example),
    the declared field type is still used. This means that, if appropriate,
    multiple protocol buffer types may use the same Python type.

    The primary implementation of this is :class:`Marshal`, which should
    usually be used instead of this class directly.
    """

    def __init__(self):
        self._rules = {}
        self._noop = NoopRule()
        self.reset()

    def register(self, proto_type: type, rule: Rule=None):
        """Register a rule against the given ``proto_type``.

        This function expects a ``proto_type`` (the descriptor class) and
        a ``rule``; an object with a ``to_python`` and ``to_proto`` method.
        Each method should return the appropriate Python or protocol buffer
        type, and be idempotent (e.g. accept either type as input).

        This function can also be used as a decorator::

            @marshal.register(timestamp_pb2.Timestamp)
            class TimestampRule:
                ...

        In this case, the class will be initialized for you with zero
        arguments.

        Args:
            proto_type (type): A protocol buffer message type.
            rule: A marshal object
        """
        if rule:
            if not isinstance(rule, Rule):
                raise TypeError('Marshal rule instances must implement `to_proto` and `to_python` methods.')
            self._rules[proto_type] = rule
            return

        def register_rule_class(rule_class: type):
            if not issubclass(rule_class, Rule):
                raise TypeError('Marshal rule subclasses must implement `to_proto` and `to_python` methods.')
            self._rules[proto_type] = rule_class()
            return rule_class
        return register_rule_class

    def reset(self):
        """Reset the registry to its initial state."""
        self._rules.clear()
        self.register(timestamp_pb2.Timestamp, dates.TimestampRule())
        self.register(duration_pb2.Duration, dates.DurationRule())
        self.register(field_mask_pb2.FieldMask, field_mask.FieldMaskRule())
        self.register(wrappers_pb2.BoolValue, wrappers.BoolValueRule())
        self.register(wrappers_pb2.BytesValue, wrappers.BytesValueRule())
        self.register(wrappers_pb2.DoubleValue, wrappers.DoubleValueRule())
        self.register(wrappers_pb2.FloatValue, wrappers.FloatValueRule())
        self.register(wrappers_pb2.Int32Value, wrappers.Int32ValueRule())
        self.register(wrappers_pb2.Int64Value, wrappers.Int64ValueRule())
        self.register(wrappers_pb2.StringValue, wrappers.StringValueRule())
        self.register(wrappers_pb2.UInt32Value, wrappers.UInt32ValueRule())
        self.register(wrappers_pb2.UInt64Value, wrappers.UInt64ValueRule())
        self.register(struct_pb2.Value, struct.ValueRule(marshal=self))
        self.register(struct_pb2.ListValue, struct.ListValueRule(marshal=self))
        self.register(struct_pb2.Struct, struct.StructRule(marshal=self))
        self.register(ProtoType.BYTES, pb_bytes.BytesRule())
        for rule_class in stringy_numbers.STRINGY_NUMBER_RULES:
            self.register(rule_class._proto_type, rule_class())

    def get_rule(self, proto_type):
        rule = self._rules.get(proto_type, self._noop)
        if rule == self._noop and hasattr(self, '_instances'):
            for _, instance in self._instances.items():
                rule = instance._rules.get(proto_type, self._noop)
                if rule != self._noop:
                    break
        return rule

    def to_python(self, proto_type, value, *, absent: bool=None):
        value_type = type(value)
        if value_type in compat.repeated_composite_types:
            return RepeatedComposite(value, marshal=self)
        if value_type in compat.repeated_scalar_types:
            if isinstance(proto_type, type):
                return RepeatedComposite(value, marshal=self, proto_type=proto_type)
            else:
                return Repeated(value, marshal=self)
        if value_type in compat.map_composite_types:
            return MapComposite(value, marshal=self)
        return self.get_rule(proto_type=proto_type).to_python(value, absent=absent)

    def to_proto(self, proto_type, value, *, strict: bool=False):
        if proto_type not in (struct_pb2.Value, struct_pb2.ListValue, struct_pb2.Struct):
            if isinstance(value, (Repeated, MapComposite)):
                return value.pb
            if isinstance(value, (list, tuple)):
                return type(value)((self.to_proto(proto_type, i) for i in value))
        if isinstance(value, dict) and (proto_type.DESCRIPTOR.has_options and proto_type.DESCRIPTOR.GetOptions().map_entry):
            recursive_type = type(proto_type().value)
            return {k: self.to_proto(recursive_type, v) for k, v in value.items()}
        pb_value = self.get_rule(proto_type=proto_type).to_proto(value)
        if strict and (not isinstance(pb_value, proto_type)):
            raise TypeError('Parameter must be instance of the same class; expected {expected}, got {got}'.format(expected=proto_type.__name__, got=pb_value.__class__.__name__))
        return pb_value