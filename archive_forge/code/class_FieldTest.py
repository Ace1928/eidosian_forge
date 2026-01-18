import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
class FieldTest(test_util.TestCase):

    def ActionOnAllFieldClasses(self, action):
        """Test all field classes except Message and Enum.

        Message and Enum require separate tests.

        Args:
          action: Callable that takes the field class as a parameter.
        """
        classes = (messages.IntegerField, messages.FloatField, messages.BooleanField, messages.BytesField, messages.StringField)
        for field_class in classes:
            action(field_class)

    def testNumberAttribute(self):
        """Test setting the number attribute."""

        def action(field_class):
            self.assertRaises(messages.InvalidNumberError, field_class, 0)
            self.assertRaises(messages.InvalidNumberError, field_class, -1)
            self.assertRaises(messages.InvalidNumberError, field_class, messages.MAX_FIELD_NUMBER + 1)
            self.assertRaises(messages.InvalidNumberError, field_class, messages.FIRST_RESERVED_FIELD_NUMBER)
            self.assertRaises(messages.InvalidNumberError, field_class, messages.LAST_RESERVED_FIELD_NUMBER)
            self.assertRaises(messages.InvalidNumberError, field_class, '1')
            field_class(number=1)
        self.ActionOnAllFieldClasses(action)

    def testRequiredAndRepeated(self):
        """Test setting the required and repeated fields."""

        def action(field_class):
            field_class(1, required=True)
            field_class(1, repeated=True)
            self.assertRaises(messages.FieldDefinitionError, field_class, 1, required=True, repeated=True)
        self.ActionOnAllFieldClasses(action)

    def testInvalidVariant(self):
        """Test field with invalid variants."""

        def action(field_class):
            if field_class is not message_types.DateTimeField:
                self.assertRaises(messages.InvalidVariantError, field_class, 1, variant=messages.Variant.ENUM)
        self.ActionOnAllFieldClasses(action)

    def testDefaultVariant(self):
        """Test that default variant is used when not set."""

        def action(field_class):
            field = field_class(1)
            self.assertEquals(field_class.DEFAULT_VARIANT, field.variant)
        self.ActionOnAllFieldClasses(action)

    def testAlternateVariant(self):
        """Test that default variant is used when not set."""
        field = messages.IntegerField(1, variant=messages.Variant.UINT32)
        self.assertEquals(messages.Variant.UINT32, field.variant)

    def testDefaultFields_Single(self):
        """Test default field is correct type (single)."""
        defaults = {messages.IntegerField: 10, messages.FloatField: 1.5, messages.BooleanField: False, messages.BytesField: b'abc', messages.StringField: u'abc'}

        def action(field_class):
            field_class(1, default=defaults[field_class])
        self.ActionOnAllFieldClasses(action)
        defaults[messages.StringField] = 'abc'
        self.ActionOnAllFieldClasses(action)

    def testStringField_BadUnicodeInDefault(self):
        """Test binary values in string field."""
        self.assertRaisesWithRegexpMatch(messages.InvalidDefaultError, "Invalid default value for StringField:.*: Field encountered non-UTF-8 string .*: 'utf.?8' codec can't decode byte 0xc3 in position 0: invalid continuation byte", messages.StringField, 1, default=b'\xc3(')

    def testDefaultFields_InvalidSingle(self):
        """Test default field is correct type (invalid single)."""

        def action(field_class):
            self.assertRaises(messages.InvalidDefaultError, field_class, 1, default=object())
        self.ActionOnAllFieldClasses(action)

    def testDefaultFields_InvalidRepeated(self):
        """Test default field does not accept defaults."""
        self.assertRaisesWithRegexpMatch(messages.FieldDefinitionError, 'Repeated fields may not have defaults', messages.StringField, 1, repeated=True, default=[1, 2, 3])

    def testDefaultFields_None(self):
        """Test none is always acceptable."""

        def action(field_class):
            field_class(1, default=None)
            field_class(1, required=True, default=None)
            field_class(1, repeated=True, default=None)
        self.ActionOnAllFieldClasses(action)

    def testDefaultFields_Enum(self):
        """Test the default for enum fields."""

        class Symbol(messages.Enum):
            ALPHA = 1
            BETA = 2
            GAMMA = 3
        field = messages.EnumField(Symbol, 1, default=Symbol.ALPHA)
        self.assertEquals(Symbol.ALPHA, field.default)

    def testDefaultFields_EnumStringDelayedResolution(self):
        """Test that enum fields resolve default strings."""
        field = messages.EnumField('apitools.base.protorpclite.descriptor.FieldDescriptor.Label', 1, default='OPTIONAL')
        self.assertEquals(descriptor.FieldDescriptor.Label.OPTIONAL, field.default)

    def testDefaultFields_EnumIntDelayedResolution(self):
        """Test that enum fields resolve default integers."""
        field = messages.EnumField('apitools.base.protorpclite.descriptor.FieldDescriptor.Label', 1, default=2)
        self.assertEquals(descriptor.FieldDescriptor.Label.REQUIRED, field.default)

    def testDefaultFields_EnumOkIfTypeKnown(self):
        """Test enum fields accept valid default values when type is known."""
        field = messages.EnumField(descriptor.FieldDescriptor.Label, 1, default='REPEATED')
        self.assertEquals(descriptor.FieldDescriptor.Label.REPEATED, field.default)

    def testDefaultFields_EnumForceCheckIfTypeKnown(self):
        """Test that enum fields validate default values if type is known."""
        self.assertRaisesWithRegexpMatch(TypeError, 'No such value for NOT_A_LABEL in Enum Label', messages.EnumField, descriptor.FieldDescriptor.Label, 1, default='NOT_A_LABEL')

    def testDefaultFields_EnumInvalidDelayedResolution(self):
        """Test that enum fields raise errors upon delayed resolution error."""
        field = messages.EnumField('apitools.base.protorpclite.descriptor.FieldDescriptor.Label', 1, default=200)
        self.assertRaisesWithRegexpMatch(TypeError, 'No such value for 200 in Enum Label', getattr, field, 'default')

    def testValidate_Valid(self):
        """Test validation of valid values."""
        values = {messages.IntegerField: 10, messages.FloatField: 1.5, messages.BooleanField: False, messages.BytesField: b'abc', messages.StringField: u'abc'}

        def action(field_class):
            field = field_class(1)
            field.validate(values[field_class])
            field = field_class(1, required=True)
            field.validate(values[field_class])
            field = field_class(1, repeated=True)
            field.validate([])
            field.validate(())
            field.validate([values[field_class]])
            field.validate((values[field_class],))
            self.assertRaises(messages.ValidationError, field.validate, values[field_class])
            self.assertRaises(messages.ValidationError, field.validate, values[field_class])
        self.ActionOnAllFieldClasses(action)

    def testValidate_Invalid(self):
        """Test validation of valid values."""
        values = {messages.IntegerField: '10', messages.FloatField: 'blah', messages.BooleanField: 0, messages.BytesField: 10.2, messages.StringField: 42}

        def action(field_class):
            field = field_class(1)
            self.assertRaises(messages.ValidationError, field.validate, values[field_class])
            field = field_class(1, required=True)
            self.assertRaises(messages.ValidationError, field.validate, values[field_class])
            field = field_class(1, repeated=True)
            self.assertRaises(messages.ValidationError, field.validate, [values[field_class]])
            self.assertRaises(messages.ValidationError, field.validate, (values[field_class],))
        self.ActionOnAllFieldClasses(action)

    def testValidate_None(self):
        """Test that None is valid for non-required fields."""

        def action(field_class):
            field = field_class(1)
            field.validate(None)
            field = field_class(1, required=True)
            self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Required field is missing', field.validate, None)
            field = field_class(1, repeated=True)
            field.validate(None)
            self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Repeated values for %s may not be None' % field_class.__name__, field.validate, [None])
            self.assertRaises(messages.ValidationError, field.validate, (None,))
        self.ActionOnAllFieldClasses(action)

    def testValidateElement(self):
        """Test validation of valid values."""
        values = {messages.IntegerField: (10, -1, 0), messages.FloatField: (1.5, -1.5, 3), messages.BooleanField: (True, False), messages.BytesField: (b'abc',), messages.StringField: (u'abc',)}

        def action(field_class):
            field = field_class(1)
            for value in values[field_class]:
                field.validate_element(value)
            field = field_class(1, required=True)
            for value in values[field_class]:
                field.validate_element(value)
            field = field_class(1, repeated=True)
            self.assertRaises(messages.ValidationError, field.validate_element, [])
            self.assertRaises(messages.ValidationError, field.validate_element, ())
            for value in values[field_class]:
                field.validate_element(value)
            self.assertRaises(messages.ValidationError, field.validate_element, list(values[field_class]))
            self.assertRaises(messages.ValidationError, field.validate_element, values[field_class])
        self.ActionOnAllFieldClasses(action)

    def testValidateCastingElement(self):
        field = messages.FloatField(1)
        self.assertEquals(type(field.validate_element(12)), float)
        self.assertEquals(type(field.validate_element(12.0)), float)
        field = messages.IntegerField(1)
        self.assertEquals(type(field.validate_element(12)), int)
        self.assertRaises(messages.ValidationError, field.validate_element, 12.0)

    def testReadOnly(self):
        """Test that objects are all read-only."""

        def action(field_class):
            field = field_class(10)
            self.assertRaises(AttributeError, setattr, field, 'number', 20)
            self.assertRaises(AttributeError, setattr, field, 'anything_else', 'whatever')
        self.ActionOnAllFieldClasses(action)

    def testMessageField(self):
        """Test the construction of message fields."""
        self.assertRaises(messages.FieldDefinitionError, messages.MessageField, str, 10)
        self.assertRaises(messages.FieldDefinitionError, messages.MessageField, messages.Message, 10)

        class MyMessage(messages.Message):
            pass
        field = messages.MessageField(MyMessage, 10)
        self.assertEquals(MyMessage, field.type)

    def testMessageField_ForwardReference(self):
        """Test the construction of forward reference message fields."""
        global MyMessage
        global ForwardMessage
        try:

            class MyMessage(messages.Message):
                self_reference = messages.MessageField('MyMessage', 1)
                forward = messages.MessageField('ForwardMessage', 2)
                nested = messages.MessageField('ForwardMessage.NestedMessage', 3)
                inner = messages.MessageField('Inner', 4)

                class Inner(messages.Message):
                    sibling = messages.MessageField('Sibling', 1)

                class Sibling(messages.Message):
                    pass

            class ForwardMessage(messages.Message):

                class NestedMessage(messages.Message):
                    pass
            self.assertEquals(MyMessage, MyMessage.field_by_name('self_reference').type)
            self.assertEquals(ForwardMessage, MyMessage.field_by_name('forward').type)
            self.assertEquals(ForwardMessage.NestedMessage, MyMessage.field_by_name('nested').type)
            self.assertEquals(MyMessage.Inner, MyMessage.field_by_name('inner').type)
            self.assertEquals(MyMessage.Sibling, MyMessage.Inner.field_by_name('sibling').type)
        finally:
            try:
                del MyMessage
                del ForwardMessage
            except:
                pass

    def testMessageField_WrongType(self):
        """Test that forward referencing the wrong type raises an error."""
        global AnEnum
        try:

            class AnEnum(messages.Enum):
                pass

            class AnotherMessage(messages.Message):
                a_field = messages.MessageField('AnEnum', 1)
            self.assertRaises(messages.FieldDefinitionError, getattr, AnotherMessage.field_by_name('a_field'), 'type')
        finally:
            del AnEnum

    def testMessageFieldValidate(self):
        """Test validation on message field."""

        class MyMessage(messages.Message):
            pass

        class AnotherMessage(messages.Message):
            pass
        field = messages.MessageField(MyMessage, 10)
        field.validate(MyMessage())
        self.assertRaises(messages.ValidationError, field.validate, AnotherMessage())

    def testMessageFieldMessageType(self):
        """Test message_type property."""

        class MyMessage(messages.Message):
            pass

        class HasMessage(messages.Message):
            field = messages.MessageField(MyMessage, 1)
        self.assertEqual(HasMessage.field.type, HasMessage.field.message_type)

    def testMessageFieldValueFromMessage(self):

        class MyMessage(messages.Message):
            pass

        class HasMessage(messages.Message):
            field = messages.MessageField(MyMessage, 1)
        instance = MyMessage()
        self.assertTrue(instance is HasMessage.field.value_from_message(instance))

    def testMessageFieldValueFromMessageWrongType(self):

        class MyMessage(messages.Message):
            pass

        class HasMessage(messages.Message):
            field = messages.MessageField(MyMessage, 1)
        self.assertRaisesWithRegexpMatch(messages.DecodeError, 'Expected type MyMessage, got int: 10', HasMessage.field.value_from_message, 10)

    def testMessageFieldValueToMessage(self):

        class MyMessage(messages.Message):
            pass

        class HasMessage(messages.Message):
            field = messages.MessageField(MyMessage, 1)
        instance = MyMessage()
        self.assertTrue(instance is HasMessage.field.value_to_message(instance))

    def testMessageFieldValueToMessageWrongType(self):

        class MyMessage(messages.Message):
            pass

        class MyOtherMessage(messages.Message):
            pass

        class HasMessage(messages.Message):
            field = messages.MessageField(MyMessage, 1)
        instance = MyOtherMessage()
        self.assertRaisesWithRegexpMatch(messages.EncodeError, 'Expected type MyMessage, got MyOtherMessage: <MyOtherMessage>', HasMessage.field.value_to_message, instance)

    def testIntegerField_AllowLong(self):
        """Test that the integer field allows for longs."""
        if six.PY2:
            messages.IntegerField(10, default=long(10))

    def testMessageFieldValidate_Initialized(self):
        """Test validation on message field."""

        class MyMessage(messages.Message):
            field1 = messages.IntegerField(1, required=True)
        field = messages.MessageField(MyMessage, 10)
        message = MyMessage()
        field.validate(message)
        message.field1 = 20
        field.validate(message)

    def testEnumField(self):
        """Test the construction of enum fields."""
        self.assertRaises(messages.FieldDefinitionError, messages.EnumField, str, 10)
        self.assertRaises(messages.FieldDefinitionError, messages.EnumField, messages.Enum, 10)

        class Color(messages.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3
        field = messages.EnumField(Color, 10)
        self.assertEquals(Color, field.type)

        class Another(messages.Enum):
            VALUE = 1
        self.assertRaises(messages.InvalidDefaultError, messages.EnumField, Color, 10, default=Another.VALUE)

    def testEnumField_ForwardReference(self):
        """Test the construction of forward reference enum fields."""
        global MyMessage
        global ForwardEnum
        global ForwardMessage
        try:

            class MyMessage(messages.Message):
                forward = messages.EnumField('ForwardEnum', 1)
                nested = messages.EnumField('ForwardMessage.NestedEnum', 2)
                inner = messages.EnumField('Inner', 3)

                class Inner(messages.Enum):
                    pass

            class ForwardEnum(messages.Enum):
                pass

            class ForwardMessage(messages.Message):

                class NestedEnum(messages.Enum):
                    pass
            self.assertEquals(ForwardEnum, MyMessage.field_by_name('forward').type)
            self.assertEquals(ForwardMessage.NestedEnum, MyMessage.field_by_name('nested').type)
            self.assertEquals(MyMessage.Inner, MyMessage.field_by_name('inner').type)
        finally:
            try:
                del MyMessage
                del ForwardEnum
                del ForwardMessage
            except:
                pass

    def testEnumField_WrongType(self):
        """Test that forward referencing the wrong type raises an error."""
        global AMessage
        try:

            class AMessage(messages.Message):
                pass

            class AnotherMessage(messages.Message):
                a_field = messages.EnumField('AMessage', 1)
            self.assertRaises(messages.FieldDefinitionError, getattr, AnotherMessage.field_by_name('a_field'), 'type')
        finally:
            del AMessage

    def testMessageDefinition(self):
        """Test that message definition is set on fields."""

        class MyMessage(messages.Message):
            my_field = messages.StringField(1)
        self.assertEquals(MyMessage, MyMessage.field_by_name('my_field').message_definition())

    def testNoneAssignment(self):
        """Test that assigning None does not change comparison."""

        class MyMessage(messages.Message):
            my_field = messages.StringField(1)
        m1 = MyMessage()
        m2 = MyMessage()
        m2.my_field = None
        self.assertEquals(m1, m2)

    def testNonUtf8Str(self):
        """Test validation fails for non-UTF-8 StringField values."""

        class Thing(messages.Message):
            string_field = messages.StringField(2)
        thing = Thing()
        self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Field string_field encountered non-UTF-8 string', setattr, thing, 'string_field', test_util.BINARY)