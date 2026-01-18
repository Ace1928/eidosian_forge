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
class MessageTest(test_util.TestCase):
    """Tests for message class."""

    def CreateMessageClass(self):
        """Creates a simple message class with 3 fields.

        Fields are defined in alphabetical order but with conflicting numeric
        order.
        """

        class ComplexMessage(messages.Message):
            a3 = messages.IntegerField(3)
            b1 = messages.StringField(1)
            c2 = messages.StringField(2)
        return ComplexMessage

    def testSameNumbers(self):
        """Test that cannot assign two fields with same numbers."""

        def action():

            class BadMessage(messages.Message):
                f1 = messages.IntegerField(1)
                f2 = messages.IntegerField(1)
        self.assertRaises(messages.DuplicateNumberError, action)

    def testStrictAssignment(self):
        """Tests that cannot assign to unknown or non-reserved attributes."""

        class SimpleMessage(messages.Message):
            field = messages.IntegerField(1)
        simple_message = SimpleMessage()
        self.assertRaises(AttributeError, setattr, simple_message, 'does_not_exist', 10)

    def testListAssignmentDoesNotCopy(self):

        class SimpleMessage(messages.Message):
            repeated = messages.IntegerField(1, repeated=True)
        message = SimpleMessage()
        original = message.repeated
        message.repeated = []
        self.assertFalse(original is message.repeated)

    def testValidate_Optional(self):
        """Tests validation of optional fields."""

        class SimpleMessage(messages.Message):
            non_required = messages.IntegerField(1)
        simple_message = SimpleMessage()
        simple_message.check_initialized()
        simple_message.non_required = 10
        simple_message.check_initialized()

    def testValidate_Required(self):
        """Tests validation of required fields."""

        class SimpleMessage(messages.Message):
            required = messages.IntegerField(1, required=True)
        simple_message = SimpleMessage()
        self.assertRaises(messages.ValidationError, simple_message.check_initialized)
        simple_message.required = 10
        simple_message.check_initialized()

    def testValidate_Repeated(self):
        """Tests validation of repeated fields."""

        class SimpleMessage(messages.Message):
            repeated = messages.IntegerField(1, repeated=True)
        simple_message = SimpleMessage()
        for valid_value in ([], [10], [10, 20], (), (10,), (10, 20)):
            simple_message.repeated = valid_value
            simple_message.check_initialized()
        simple_message.repeated = []
        simple_message.check_initialized()
        for invalid_value in (10, ['10', '20'], [None], (None,)):
            self.assertRaises(messages.ValidationError, setattr, simple_message, 'repeated', invalid_value)

    def testIsInitialized(self):
        """Tests is_initialized."""

        class SimpleMessage(messages.Message):
            required = messages.IntegerField(1, required=True)
        simple_message = SimpleMessage()
        self.assertFalse(simple_message.is_initialized())
        simple_message.required = 10
        self.assertTrue(simple_message.is_initialized())

    def testIsInitializedNestedField(self):
        """Tests is_initialized for nested fields."""

        class SimpleMessage(messages.Message):
            required = messages.IntegerField(1, required=True)

        class NestedMessage(messages.Message):
            simple = messages.MessageField(SimpleMessage, 1)
        simple_message = SimpleMessage()
        self.assertFalse(simple_message.is_initialized())
        nested_message = NestedMessage(simple=simple_message)
        self.assertFalse(nested_message.is_initialized())
        simple_message.required = 10
        self.assertTrue(simple_message.is_initialized())
        self.assertTrue(nested_message.is_initialized())

    def testInitializeNestedFieldFromDict(self):
        """Tests initializing nested fields from dict."""

        class SimpleMessage(messages.Message):
            required = messages.IntegerField(1, required=True)

        class NestedMessage(messages.Message):
            simple = messages.MessageField(SimpleMessage, 1)

        class RepeatedMessage(messages.Message):
            simple = messages.MessageField(SimpleMessage, 1, repeated=True)
        nested_message1 = NestedMessage(simple={'required': 10})
        self.assertTrue(nested_message1.is_initialized())
        self.assertTrue(nested_message1.simple.is_initialized())
        nested_message2 = NestedMessage()
        nested_message2.simple = {'required': 10}
        self.assertTrue(nested_message2.is_initialized())
        self.assertTrue(nested_message2.simple.is_initialized())
        repeated_values = [{}, {'required': 10}, SimpleMessage(required=20)]
        repeated_message1 = RepeatedMessage(simple=repeated_values)
        self.assertEquals(3, len(repeated_message1.simple))
        self.assertFalse(repeated_message1.is_initialized())
        repeated_message1.simple[0].required = 0
        self.assertTrue(repeated_message1.is_initialized())
        repeated_message2 = RepeatedMessage()
        repeated_message2.simple = repeated_values
        self.assertEquals(3, len(repeated_message2.simple))
        self.assertFalse(repeated_message2.is_initialized())
        repeated_message2.simple[0].required = 0
        self.assertTrue(repeated_message2.is_initialized())

    def testNestedMethodsNotAllowed(self):
        """Test that method definitions on Message classes are not allowed."""

        def action():

            class WithMethods(messages.Message):

                def not_allowed(self):
                    pass
        self.assertRaises(messages.MessageDefinitionError, action)

    def testNestedAttributesNotAllowed(self):
        """Test attribute assignment on Message classes is not allowed."""

        def int_attribute():

            class WithMethods(messages.Message):
                not_allowed = 1

        def string_attribute():

            class WithMethods(messages.Message):
                not_allowed = 'not allowed'

        def enum_attribute():

            class WithMethods(messages.Message):
                not_allowed = Color.RED
        for action in (int_attribute, string_attribute, enum_attribute):
            self.assertRaises(messages.MessageDefinitionError, action)

    def testNameIsSetOnFields(self):
        """Make sure name is set on fields after Message class init."""

        class HasNamedFields(messages.Message):
            field = messages.StringField(1)
        self.assertEquals('field', HasNamedFields.field_by_number(1).name)

    def testSubclassingMessageDisallowed(self):
        """Not permitted to create sub-classes of message classes."""

        class SuperClass(messages.Message):
            pass

        def action():

            class SubClass(SuperClass):
                pass
        self.assertRaises(messages.MessageDefinitionError, action)

    def testAllFields(self):
        """Test all_fields method."""
        ComplexMessage = self.CreateMessageClass()
        fields = list(ComplexMessage.all_fields())
        fields = sorted(fields, key=lambda f: f.name)
        self.assertEquals(3, len(fields))
        self.assertEquals('a3', fields[0].name)
        self.assertEquals('b1', fields[1].name)
        self.assertEquals('c2', fields[2].name)

    def testFieldByName(self):
        """Test getting field by name."""
        ComplexMessage = self.CreateMessageClass()
        self.assertEquals(3, ComplexMessage.field_by_name('a3').number)
        self.assertEquals(1, ComplexMessage.field_by_name('b1').number)
        self.assertEquals(2, ComplexMessage.field_by_name('c2').number)
        self.assertRaises(KeyError, ComplexMessage.field_by_name, 'unknown')

    def testFieldByNumber(self):
        """Test getting field by number."""
        ComplexMessage = self.CreateMessageClass()
        self.assertEquals('a3', ComplexMessage.field_by_number(3).name)
        self.assertEquals('b1', ComplexMessage.field_by_number(1).name)
        self.assertEquals('c2', ComplexMessage.field_by_number(2).name)
        self.assertRaises(KeyError, ComplexMessage.field_by_number, 4)

    def testGetAssignedValue(self):
        """Test getting the assigned value of a field."""

        class SomeMessage(messages.Message):
            a_value = messages.StringField(1, default=u'a default')
        message = SomeMessage()
        self.assertEquals(None, message.get_assigned_value('a_value'))
        message.a_value = u'a string'
        self.assertEquals(u'a string', message.get_assigned_value('a_value'))
        message.a_value = u'a default'
        self.assertEquals(u'a default', message.get_assigned_value('a_value'))
        self.assertRaisesWithRegexpMatch(AttributeError, 'Message SomeMessage has no field no_such_field', message.get_assigned_value, 'no_such_field')

    def testReset(self):
        """Test resetting a field value."""

        class SomeMessage(messages.Message):
            a_value = messages.StringField(1, default=u'a default')
            repeated = messages.IntegerField(2, repeated=True)
        message = SomeMessage()
        self.assertRaises(AttributeError, message.reset, 'unknown')
        self.assertEquals(u'a default', message.a_value)
        message.reset('a_value')
        self.assertEquals(u'a default', message.a_value)
        message.a_value = u'a new value'
        self.assertEquals(u'a new value', message.a_value)
        message.reset('a_value')
        self.assertEquals(u'a default', message.a_value)
        message.repeated = [1, 2, 3]
        self.assertEquals([1, 2, 3], message.repeated)
        saved = message.repeated
        message.reset('repeated')
        self.assertEquals([], message.repeated)
        self.assertIsInstance(message.repeated, messages.FieldList)
        self.assertEquals([1, 2, 3], saved)

    def testAllowNestedEnums(self):
        """Test allowing nested enums in a message definition."""

        class Trade(messages.Message):

            class Duration(messages.Enum):
                GTC = 1
                DAY = 2

            class Currency(messages.Enum):
                USD = 1
                GBP = 2
                INR = 3
        self.assertEquals(['Currency', 'Duration'], Trade.__enums__)
        self.assertEquals(Trade, Trade.Duration.message_definition())

    def testAllowNestedMessages(self):
        """Test allowing nested messages in a message definition."""

        class Trade(messages.Message):

            class Lot(messages.Message):
                pass

            class Agent(messages.Message):
                pass
        self.assertEquals(['Agent', 'Lot'], Trade.__messages__)
        self.assertEquals(Trade, Trade.Agent.message_definition())
        self.assertEquals(Trade, Trade.Lot.message_definition())

        def action():

            class Trade(messages.Message):
                NiceTry = messages.Message
        self.assertRaises(messages.MessageDefinitionError, action)

    def testDisallowClassAssignments(self):
        """Test setting class attributes may not happen."""

        class MyMessage(messages.Message):
            pass
        self.assertRaises(AttributeError, setattr, MyMessage, 'x', 'do not assign')

    def testEquality(self):
        """Test message class equality."""

        class MyEnum(messages.Enum):
            val1 = 1
            val2 = 2

        class AnotherMessage(messages.Message):
            string = messages.StringField(1)

        class MyMessage(messages.Message):
            field1 = messages.IntegerField(1)
            field2 = messages.EnumField(MyEnum, 2)
            field3 = messages.MessageField(AnotherMessage, 3)
        message1 = MyMessage()
        self.assertNotEquals('hi', message1)
        self.assertNotEquals(AnotherMessage(), message1)
        self.assertEquals(message1, message1)
        message2 = MyMessage()
        self.assertEquals(message1, message2)
        message1.field1 = 10
        self.assertNotEquals(message1, message2)
        message2.field1 = 20
        self.assertNotEquals(message1, message2)
        message2.field1 = 10
        self.assertEquals(message1, message2)
        message1.field2 = MyEnum.val1
        self.assertNotEquals(message1, message2)
        message2.field2 = MyEnum.val2
        self.assertNotEquals(message1, message2)
        message2.field2 = MyEnum.val1
        self.assertEquals(message1, message2)
        message1.field3 = AnotherMessage()
        message1.field3.string = 'value1'
        self.assertNotEquals(message1, message2)
        message2.field3 = AnotherMessage()
        message2.field3.string = 'value2'
        self.assertNotEquals(message1, message2)
        message2.field3.string = 'value1'
        self.assertEquals(message1, message2)

    def testEqualityWithUnknowns(self):
        """Test message class equality with unknown fields."""

        class MyMessage(messages.Message):
            field1 = messages.IntegerField(1)
        message1 = MyMessage()
        message2 = MyMessage()
        self.assertEquals(message1, message2)
        message1.set_unrecognized_field('unknown1', 'value1', messages.Variant.STRING)
        self.assertEquals(message1, message2)
        message1.set_unrecognized_field('unknown2', ['asdf', 3], messages.Variant.STRING)
        message1.set_unrecognized_field('unknown3', 4.7, messages.Variant.DOUBLE)
        self.assertEquals(message1, message2)

    def testUnrecognizedFieldInvalidVariant(self):

        class MyMessage(messages.Message):
            field1 = messages.IntegerField(1)
        message1 = MyMessage()
        self.assertRaises(TypeError, message1.set_unrecognized_field, 'unknown4', {'unhandled': 'type'}, None)
        self.assertRaises(TypeError, message1.set_unrecognized_field, 'unknown4', {'unhandled': 'type'}, 123)

    def testRepr(self):
        """Test represtation of Message object."""

        class MyMessage(messages.Message):
            integer_value = messages.IntegerField(1)
            string_value = messages.StringField(2)
            unassigned = messages.StringField(3)
            unassigned_with_default = messages.StringField(4, default=u'a default')
        my_message = MyMessage()
        my_message.integer_value = 42
        my_message.string_value = u'A string'
        pat = re.compile("<MyMessage\\n integer_value: 42\\n string_value: [u]?'A string'>")
        self.assertTrue(pat.match(repr(my_message)) is not None)

    def testValidation(self):
        """Test validation of message values."""

        class SubMessage(messages.Message):
            pass

        class Message(messages.Message):
            val = messages.MessageField(SubMessage, 1)
        message = Message()
        message_field = messages.MessageField(Message, 1)
        message_field.validate(message)
        message.val = SubMessage()
        message_field.validate(message)
        self.assertRaises(messages.ValidationError, setattr, message, 'val', [SubMessage()])

        class Message(messages.Message):
            val = messages.MessageField(SubMessage, 1, required=True)
        message = Message()
        message_field = messages.MessageField(Message, 1)
        message_field.validate(message)
        message.val = SubMessage()
        message_field.validate(message)
        self.assertRaises(messages.ValidationError, setattr, message, 'val', [SubMessage()])

        class Message(messages.Message):
            val = messages.MessageField(SubMessage, 1, repeated=True)
        message = Message()
        message_field = messages.MessageField(Message, 1)
        message_field.validate(message)
        self.assertRaisesWithRegexpMatch(messages.ValidationError, 'Field val is repeated. Found: <SubMessage>', setattr, message, 'val', SubMessage())
        message.val = [SubMessage()]
        message_field.validate(message)

    def testDefinitionName(self):
        """Test message name."""

        class MyMessage(messages.Message):
            pass
        module_name = test_util.get_module_name(FieldTest)
        self.assertEquals('%s.MyMessage' % module_name, MyMessage.definition_name())
        self.assertEquals(module_name, MyMessage.outer_definition_name())
        self.assertEquals(module_name, MyMessage.definition_package())
        self.assertEquals(six.text_type, type(MyMessage.definition_name()))
        self.assertEquals(six.text_type, type(MyMessage.outer_definition_name()))
        self.assertEquals(six.text_type, type(MyMessage.definition_package()))

    def testDefinitionName_OverrideModule(self):
        """Test message module is overriden by module package name."""

        class MyMessage(messages.Message):
            pass
        global package
        package = 'my.package'
        try:
            self.assertEquals('my.package.MyMessage', MyMessage.definition_name())
            self.assertEquals('my.package', MyMessage.outer_definition_name())
            self.assertEquals('my.package', MyMessage.definition_package())
            self.assertEquals(six.text_type, type(MyMessage.definition_name()))
            self.assertEquals(six.text_type, type(MyMessage.outer_definition_name()))
            self.assertEquals(six.text_type, type(MyMessage.definition_package()))
        finally:
            del package

    def testDefinitionName_NoModule(self):
        """Test what happens when there is no module for message."""

        class MyMessage(messages.Message):
            pass
        original_modules = sys.modules
        sys.modules = dict(sys.modules)
        try:
            del sys.modules[__name__]
            self.assertEquals('MyMessage', MyMessage.definition_name())
            self.assertEquals(None, MyMessage.outer_definition_name())
            self.assertEquals(None, MyMessage.definition_package())
            self.assertEquals(six.text_type, type(MyMessage.definition_name()))
        finally:
            sys.modules = original_modules

    def testDefinitionName_Nested(self):
        """Test nested message names."""

        class MyMessage(messages.Message):

            class NestedMessage(messages.Message):

                class NestedMessage(messages.Message):
                    pass
        module_name = test_util.get_module_name(MessageTest)
        self.assertEquals('%s.MyMessage.NestedMessage' % module_name, MyMessage.NestedMessage.definition_name())
        self.assertEquals('%s.MyMessage' % module_name, MyMessage.NestedMessage.outer_definition_name())
        self.assertEquals(module_name, MyMessage.NestedMessage.definition_package())
        self.assertEquals('%s.MyMessage.NestedMessage.NestedMessage' % module_name, MyMessage.NestedMessage.NestedMessage.definition_name())
        self.assertEquals('%s.MyMessage.NestedMessage' % module_name, MyMessage.NestedMessage.NestedMessage.outer_definition_name())
        self.assertEquals(module_name, MyMessage.NestedMessage.NestedMessage.definition_package())

    def testMessageDefinition(self):
        """Test that enumeration knows its enclosing message definition."""

        class OuterMessage(messages.Message):

            class InnerMessage(messages.Message):
                pass
        self.assertEquals(None, OuterMessage.message_definition())
        self.assertEquals(OuterMessage, OuterMessage.InnerMessage.message_definition())

    def testConstructorKwargs(self):
        """Test kwargs via constructor."""

        class SomeMessage(messages.Message):
            name = messages.StringField(1)
            number = messages.IntegerField(2)
        expected = SomeMessage()
        expected.name = 'my name'
        expected.number = 200
        self.assertEquals(expected, SomeMessage(name='my name', number=200))

    def testConstructorNotAField(self):
        """Test kwargs via constructor with wrong names."""

        class SomeMessage(messages.Message):
            pass
        self.assertRaisesWithRegexpMatch(AttributeError, 'May not assign arbitrary value does_not_exist to message SomeMessage', SomeMessage, does_not_exist=10)

    def testGetUnsetRepeatedValue(self):

        class SomeMessage(messages.Message):
            repeated = messages.IntegerField(1, repeated=True)
        instance = SomeMessage()
        self.assertEquals([], instance.repeated)
        self.assertTrue(isinstance(instance.repeated, messages.FieldList))

    def testCompareAutoInitializedRepeatedFields(self):

        class SomeMessage(messages.Message):
            repeated = messages.IntegerField(1, repeated=True)
        message1 = SomeMessage(repeated=[])
        message2 = SomeMessage()
        self.assertEquals(message1, message2)

    def testUnknownValues(self):
        """Test message class equality with unknown fields."""

        class MyMessage(messages.Message):
            field1 = messages.IntegerField(1)
        message = MyMessage()
        self.assertEquals([], message.all_unrecognized_fields())
        self.assertEquals((None, None), message.get_unrecognized_field_info('doesntexist'))
        self.assertEquals((None, None), message.get_unrecognized_field_info('doesntexist', None, None))
        self.assertEquals(('defaultvalue', 'defaultwire'), message.get_unrecognized_field_info('doesntexist', 'defaultvalue', 'defaultwire'))
        self.assertEquals((3, None), message.get_unrecognized_field_info('doesntexist', value_default=3))
        message.set_unrecognized_field('exists', 9.5, messages.Variant.DOUBLE)
        self.assertEquals(1, len(message.all_unrecognized_fields()))
        self.assertTrue('exists' in message.all_unrecognized_fields())
        self.assertEquals((9.5, messages.Variant.DOUBLE), message.get_unrecognized_field_info('exists'))
        self.assertEquals((9.5, messages.Variant.DOUBLE), message.get_unrecognized_field_info('exists', 'type', 1234))
        self.assertEquals((1234, None), message.get_unrecognized_field_info('doesntexist', 1234))
        message.set_unrecognized_field('another', 'value', messages.Variant.STRING)
        self.assertEquals(2, len(message.all_unrecognized_fields()))
        self.assertTrue('exists' in message.all_unrecognized_fields())
        self.assertTrue('another' in message.all_unrecognized_fields())
        self.assertEquals((9.5, messages.Variant.DOUBLE), message.get_unrecognized_field_info('exists'))
        self.assertEquals(('value', messages.Variant.STRING), message.get_unrecognized_field_info('another'))
        message.set_unrecognized_field('typetest1', ['list', 0, ('test',)], messages.Variant.STRING)
        self.assertEquals((['list', 0, ('test',)], messages.Variant.STRING), message.get_unrecognized_field_info('typetest1'))
        message.set_unrecognized_field('typetest2', '', messages.Variant.STRING)
        self.assertEquals(('', messages.Variant.STRING), message.get_unrecognized_field_info('typetest2'))

    def testPickle(self):
        """Testing pickling and unpickling of Message instances."""
        global MyEnum
        global AnotherMessage
        global MyMessage

        class MyEnum(messages.Enum):
            val1 = 1
            val2 = 2

        class AnotherMessage(messages.Message):
            string = messages.StringField(1, repeated=True)

        class MyMessage(messages.Message):
            field1 = messages.IntegerField(1)
            field2 = messages.EnumField(MyEnum, 2)
            field3 = messages.MessageField(AnotherMessage, 3)
        message = MyMessage(field1=1, field2=MyEnum.val2, field3=AnotherMessage(string=['a', 'b', 'c']))
        message.set_unrecognized_field('exists', 'value', messages.Variant.STRING)
        message.set_unrecognized_field('repeated', ['list', 0, ('test',)], messages.Variant.STRING)
        unpickled = pickle.loads(pickle.dumps(message))
        self.assertEquals(message, unpickled)
        self.assertTrue(AnotherMessage.string is unpickled.field3.string.field)
        self.assertTrue('exists' in message.all_unrecognized_fields())
        self.assertEquals(('value', messages.Variant.STRING), message.get_unrecognized_field_info('exists'))
        self.assertEquals((['list', 0, ('test',)], messages.Variant.STRING), message.get_unrecognized_field_info('repeated'))