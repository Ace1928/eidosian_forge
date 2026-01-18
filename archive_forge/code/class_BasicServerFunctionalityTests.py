import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class BasicServerFunctionalityTests(IRCTestCase):

    def setUp(self):
        self.f = StringIOWithoutClosing()
        self.t = protocol.FileWrapper(self.f)
        self.p = irc.IRC()
        self.p.makeConnection(self.t)

    def check(self, s):
        """
        Make sure that the internal buffer equals a specified value.

        @param s: the value to compare against buffer
        @type s: L{bytes} or L{unicode}
        """
        bufferValue = self.f.getvalue()
        if isinstance(s, str):
            bufferValue = bufferValue.decode('utf-8')
        self.assertEqual(bufferValue, s)

    def test_sendMessage(self):
        """
        Passing a command and parameters to L{IRC.sendMessage} results in a
        query string that consists of the command and parameters, separated by
        a space, ending with '\r
'.
        """
        self.p.sendMessage('CMD', 'param1', 'param2')
        self.check('CMD param1 param2\r\n')

    def test_sendCommand(self):
        """
        Passing a command and parameters to L{IRC.sendCommand} results in a
        query string that consists of the command and parameters, separated by
        a space, ending with '\r
'.

        The format is described in more detail in
        U{RFC 1459 <https://tools.ietf.org/html/rfc1459.html#section-2.3>}.
        """
        self.p.sendCommand('CMD', ('param1', 'param2'))
        self.check('CMD param1 param2\r\n')

    def test_sendUnicodeCommand(self):
        """
        Passing unicode parameters to L{IRC.sendCommand} encodes the parameters
        in UTF-8.
        """
        self.p.sendCommand('CMD', ('param¹', 'param²'))
        self.check(b'CMD param\xc2\xb9 param\xc2\xb2\r\n')

    def test_sendMessageNoCommand(self):
        """
        Passing L{None} as the command to L{IRC.sendMessage} raises a
        C{ValueError}.
        """
        error = self.assertRaises(ValueError, self.p.sendMessage, None, 'param1', 'param2')
        self.assertEqual(str(error), 'IRC message requires a command.')

    def test_sendCommandNoCommand(self):
        """
        Passing L{None} as the command to L{IRC.sendCommand} raises a
        C{ValueError}.
        """
        error = self.assertRaises(ValueError, self.p.sendCommand, None, ('param1', 'param2'))
        self.assertEqual(error.args[0], 'IRC message requires a command.')

    def test_sendMessageInvalidCommand(self):
        """
        Passing an invalid string command to L{IRC.sendMessage} raises a
        C{ValueError}.
        """
        error = self.assertRaises(ValueError, self.p.sendMessage, ' ', 'param1', 'param2')
        self.assertEqual(str(error), "Somebody screwed up, 'cuz this doesn't look like a command to me:  ")

    def test_sendCommandInvalidCommand(self):
        """
        Passing an invalid string command to L{IRC.sendCommand} raises a
        C{ValueError}.
        """
        error = self.assertRaises(ValueError, self.p.sendCommand, ' ', ('param1', 'param2'))
        self.assertEqual(error.args[0], 'Invalid command: " "')

    def test_sendCommandWithPrefix(self):
        """
        Passing a command and parameters with a specified prefix to
        L{IRC.sendCommand} results in a proper query string including the
        specified line prefix.
        """
        self.p.sendCommand('CMD', ('param1', 'param2'), 'irc.example.com')
        self.check(b':irc.example.com CMD param1 param2\r\n')

    def test_sendCommandWithTags(self):
        """
        Passing a command and parameters with a specified prefix and tags
        to L{IRC.sendCommand} results in a proper query string including the
        specified line prefix and appropriate tags syntax.  The query string
        should be output as follows:
        @tags :prefix COMMAND param1 param2\r

        The tags are a string of IRCv3 tags, preceded by '@'.  The rest
        of the string is as described in test_sendMessage.  For more on
        the message tag format, see U{the IRCv3 specification
        <https://ircv3.net/specs/core/message-tags-3.2.html>}.
        """
        sendTags = {'aaa': 'bbb', 'ccc': None, 'example.com/ddd': 'eee'}
        expectedTags = (b'aaa=bbb', b'ccc', b'example.com/ddd=eee')
        self.p.sendCommand('CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        outMsg = self.f.getvalue()
        outTagStr, outLine = outMsg.split(b' ', 1)
        outTags = outTagStr[1:].split(b';')
        self.assertEqual(outLine, b':irc.example.com CMD param1 param2\r\n')
        self.assertEqual(sorted(expectedTags), sorted(outTags))

    def test_sendCommandValidateEmptyTags(self):
        """
        Passing empty tag names to L{IRC.sendCommand} raises a C{ValueError}.
        """
        sendTags = {'aaa': 'bbb', 'ccc': None, '': ''}
        error = self.assertRaises(ValueError, self.p.sendCommand, 'CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        self.assertEqual(error.args[0], 'A tag name is required.')

    def test_sendCommandValidateNoneTags(self):
        """
        Passing None as a tag name to L{IRC.sendCommand} raises a
        C{ValueError}.
        """
        sendTags = {'aaa': 'bbb', 'ccc': None, None: 'beep'}
        error = self.assertRaises(ValueError, self.p.sendCommand, 'CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        self.assertEqual(error.args[0], 'A tag name is required.')

    def test_sendCommandValidateTagsWithSpaces(self):
        """
        Passing a tag name containing spaces to L{IRC.sendCommand} raises a
        C{ValueError}.
        """
        sendTags = {'aaa bbb': 'ccc'}
        error = self.assertRaises(ValueError, self.p.sendCommand, 'CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        self.assertEqual(error.args[0], 'Tag contains invalid characters.')

    def test_sendCommandValidateTagsWithInvalidChars(self):
        """
        Passing a tag name containing invalid characters to L{IRC.sendCommand}
        raises a C{ValueError}.
        """
        sendTags = {'aaa_b^@': 'ccc'}
        error = self.assertRaises(ValueError, self.p.sendCommand, 'CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        self.assertEqual(error.args[0], 'Tag contains invalid characters.')

    def test_sendCommandValidateTagValueEscaping(self):
        """
        Tags with values containing invalid characters passed to
        L{IRC.sendCommand} are escaped.
        """
        sendTags = {'aaa': 'bbb', 'ccc': 'test\r\n \\;;'}
        expectedTags = (b'aaa=bbb', b'ccc=test\\r\\n\\s\\\\\\:\\:')
        self.p.sendCommand('CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        outMsg = self.f.getvalue()
        outTagStr, outLine = outMsg.split(b' ', 1)
        outTags = outTagStr[1:].split(b';')
        self.assertEqual(sorted(outTags), sorted(expectedTags))

    def testPrivmsg(self):
        self.p.privmsg('this-is-sender', 'this-is-recip', 'this is message')
        self.check(':this-is-sender PRIVMSG this-is-recip :this is message\r\n')

    def testNotice(self):
        self.p.notice('this-is-sender', 'this-is-recip', 'this is notice')
        self.check(':this-is-sender NOTICE this-is-recip :this is notice\r\n')

    def testAction(self):
        self.p.action('this-is-sender', 'this-is-recip', 'this is action')
        self.check(':this-is-sender ACTION this-is-recip :this is action\r\n')

    def testJoin(self):
        self.p.join('this-person', '#this-channel')
        self.check(':this-person JOIN #this-channel\r\n')

    def testPart(self):
        self.p.part('this-person', '#that-channel')
        self.check(':this-person PART #that-channel\r\n')

    def testWhois(self):
        """
        Verify that a whois by the client receives the right protocol actions
        from the server.
        """
        timestamp = int(time.time() - 100)
        hostname = self.p.hostname
        req = 'requesting-nick'
        targ = 'target-nick'
        self.p.whois(req, targ, 'target', 'host.com', 'Target User', 'irc.host.com', 'A fake server', False, 12, timestamp, ['#fakeusers', '#fakemisc'])
        lines = [':%(hostname)s 311 %(req)s %(targ)s target host.com * :Target User', ':%(hostname)s 312 %(req)s %(targ)s irc.host.com :A fake server', ':%(hostname)s 317 %(req)s %(targ)s 12 %(timestamp)s :seconds idle, signon time', ':%(hostname)s 319 %(req)s %(targ)s :#fakeusers #fakemisc', ':%(hostname)s 318 %(req)s %(targ)s :End of WHOIS list.', '']
        expected = '\r\n'.join(lines) % dict(hostname=hostname, timestamp=timestamp, req=req, targ=targ)
        self.check(expected)