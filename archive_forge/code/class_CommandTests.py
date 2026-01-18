import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class CommandTests(TestCase):
    """
    Tests for L{amp.Argument} and L{amp.Command}.
    """

    def test_argumentInterface(self):
        """
        L{Argument} instances provide L{amp.IArgumentType}.
        """
        self.assertTrue(verifyObject(amp.IArgumentType, amp.Argument()))

    def test_parseResponse(self):
        """
        There should be a class method of Command which accepts a
        mapping of argument names to serialized forms and returns a
        similar mapping whose values have been parsed via the
        Command's response schema.
        """
        protocol = object()
        result = b'whatever'
        strings = {b'weird': result}
        self.assertEqual(ProtocolIncludingCommand.parseResponse(strings, protocol), {'weird': (result, protocol)})

    def test_callRemoteCallsParseResponse(self):
        """
        Making a remote call on a L{amp.Command} subclass which
        overrides the C{parseResponse} method should call that
        C{parseResponse} method to get the response.
        """
        client = NoNetworkProtocol()
        thingy = b'weeoo'
        response = client.callRemote(MagicSchemaCommand, weird=thingy)

        def gotResponse(ign):
            self.assertEqual(client.parseResponseArguments, ({'weird': thingy}, client))
        response.addCallback(gotResponse)
        return response

    def test_parseArguments(self):
        """
        There should be a class method of L{amp.Command} which accepts
        a mapping of argument names to serialized forms and returns a
        similar mapping whose values have been parsed via the
        command's argument schema.
        """
        protocol = object()
        result = b'whatever'
        strings = {b'weird': result}
        self.assertEqual(ProtocolIncludingCommand.parseArguments(strings, protocol), {'weird': (result, protocol)})

    def test_responderCallsParseArguments(self):
        """
        Making a remote call on a L{amp.Command} subclass which
        overrides the C{parseArguments} method should call that
        C{parseArguments} method to get the arguments.
        """
        protocol = NoNetworkProtocol()
        responder = protocol.locateResponder(MagicSchemaCommand.commandName)
        argument = object()
        response = responder(dict(weird=argument))
        response.addCallback(lambda ign: self.assertEqual(protocol.parseArgumentsArguments, ({'weird': argument}, protocol)))
        return response

    def test_makeArguments(self):
        """
        There should be a class method of L{amp.Command} which accepts
        a mapping of argument names to objects and returns a similar
        mapping whose values have been serialized via the command's
        argument schema.
        """
        protocol = object()
        argument = object()
        objects = {'weird': argument}
        ident = '%d:%d' % (id(argument), id(protocol))
        self.assertEqual(ProtocolIncludingCommand.makeArguments(objects, protocol), {b'weird': ident.encode('ascii')})

    def test_makeArgumentsUsesCommandType(self):
        """
        L{amp.Command.makeArguments}'s return type should be the type
        of the result of L{amp.Command.commandType}.
        """
        protocol = object()
        objects = {'weird': b'whatever'}
        result = ProtocolIncludingCommandWithDifferentCommandType.makeArguments(objects, protocol)
        self.assertIs(type(result), MyBox)

    def test_callRemoteCallsMakeArguments(self):
        """
        Making a remote call on a L{amp.Command} subclass which
        overrides the C{makeArguments} method should call that
        C{makeArguments} method to get the response.
        """
        client = NoNetworkProtocol()
        argument = object()
        response = client.callRemote(MagicSchemaCommand, weird=argument)

        def gotResponse(ign):
            self.assertEqual(client.makeArgumentsArguments, ({'weird': argument}, client))
        response.addCallback(gotResponse)
        return response

    def test_extraArgumentsDisallowed(self):
        """
        L{Command.makeArguments} raises L{amp.InvalidSignature} if the objects
        dictionary passed to it includes a key which does not correspond to the
        Python identifier for a defined argument.
        """
        self.assertRaises(amp.InvalidSignature, Hello.makeArguments, dict(hello='hello', bogusArgument=object()), None)

    def test_wireSpellingDisallowed(self):
        """
        If a command argument conflicts with a Python keyword, the
        untransformed argument name is not allowed as a key in the dictionary
        passed to L{Command.makeArguments}.  If it is supplied,
        L{amp.InvalidSignature} is raised.

        This may be a pointless implementation restriction which may be lifted.
        The current behavior is tested to verify that such arguments are not
        silently dropped on the floor (the previous behavior).
        """
        self.assertRaises(amp.InvalidSignature, Hello.makeArguments, dict(hello='required', **{'print': 'print value'}), None)

    def test_commandNameDefaultsToClassNameAsByteString(self):
        """
        A L{Command} subclass without a defined C{commandName} that's
        not a byte string.
        """

        class NewCommand(amp.Command):
            """
            A new command.
            """
        self.assertEqual(b'NewCommand', NewCommand.commandName)

    def test_commandNameMustBeAByteString(self):
        """
        A L{Command} subclass cannot be defined with a C{commandName} that's
        not a byte string.
        """
        error = self.assertRaises(TypeError, type, 'NewCommand', (amp.Command,), {'commandName': 'FOO'})
        self.assertRegex(str(error), "^Command names must be byte strings, got: u?'FOO'$")

    def test_commandArgumentsMustBeNamedWithByteStrings(self):
        """
        A L{Command} subclass's C{arguments} must have byte string names.
        """
        error = self.assertRaises(TypeError, type, 'NewCommand', (amp.Command,), {'arguments': [('foo', None)]})
        self.assertRegex(str(error), "^Argument names must be byte strings, got: u?'foo'$")

    def test_commandResponseMustBeNamedWithByteStrings(self):
        """
        A L{Command} subclass's C{response} must have byte string names.
        """
        error = self.assertRaises(TypeError, type, 'NewCommand', (amp.Command,), {'response': [('foo', None)]})
        self.assertRegex(str(error), "^Response names must be byte strings, got: u?'foo'$")

    def test_commandErrorsIsConvertedToDict(self):
        """
        A L{Command} subclass's C{errors} is coerced into a C{dict}.
        """

        class NewCommand(amp.Command):
            errors = [(ZeroDivisionError, b'ZDE')]
        self.assertEqual({ZeroDivisionError: b'ZDE'}, NewCommand.errors)

    def test_commandErrorsMustUseBytesForOnWireRepresentation(self):
        """
        A L{Command} subclass's C{errors} must map exceptions to byte strings.
        """
        error = self.assertRaises(TypeError, type, 'NewCommand', (amp.Command,), {'errors': [(ZeroDivisionError, 'foo')]})
        self.assertRegex(str(error), "^Error names must be byte strings, got: u?'foo'$")

    def test_commandFatalErrorsIsConvertedToDict(self):
        """
        A L{Command} subclass's C{fatalErrors} is coerced into a C{dict}.
        """

        class NewCommand(amp.Command):
            fatalErrors = [(ZeroDivisionError, b'ZDE')]
        self.assertEqual({ZeroDivisionError: b'ZDE'}, NewCommand.fatalErrors)

    def test_commandFatalErrorsMustUseBytesForOnWireRepresentation(self):
        """
        A L{Command} subclass's C{fatalErrors} must map exceptions to byte
        strings.
        """
        error = self.assertRaises(TypeError, type, 'NewCommand', (amp.Command,), {'fatalErrors': [(ZeroDivisionError, 'foo')]})
        self.assertRegex(str(error), "^Fatal error names must be byte strings, got: u?'foo'$")