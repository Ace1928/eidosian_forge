import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class ServerSupportedFeatureTests(IRCTestCase):
    """
    Tests for L{ServerSupportedFeatures} and related functions.
    """

    def test_intOrDefault(self):
        """
        L{_intOrDefault} converts values to C{int} if possible, otherwise
        returns a default value.
        """
        self.assertEqual(irc._intOrDefault(None), None)
        self.assertEqual(irc._intOrDefault([]), None)
        self.assertEqual(irc._intOrDefault(''), None)
        self.assertEqual(irc._intOrDefault('hello', 5), 5)
        self.assertEqual(irc._intOrDefault('123'), 123)
        self.assertEqual(irc._intOrDefault(123), 123)

    def test_splitParam(self):
        """
        L{ServerSupportedFeatures._splitParam} splits ISUPPORT parameters
        into key and values. Parameters without a separator are split into a
        key and a list containing only the empty string. Escaped parameters
        are unescaped.
        """
        params = [('FOO', ('FOO', [''])), ('FOO=', ('FOO', [''])), ('FOO=1', ('FOO', ['1'])), ('FOO=1,2,3', ('FOO', ['1', '2', '3'])), ('FOO=A\\x20B', ('FOO', ['A B'])), ('FOO=\\x5Cx', ('FOO', ['\\x'])), ('FOO=\\', ('FOO', ['\\'])), ('FOO=\\n', ('FOO', ['\\n']))]
        _splitParam = irc.ServerSupportedFeatures._splitParam
        for param, expected in params:
            res = _splitParam(param)
            self.assertEqual(res, expected)
        self.assertRaises(ValueError, _splitParam, 'FOO=\\x')
        self.assertRaises(ValueError, _splitParam, 'FOO=\\xNN')
        self.assertRaises(ValueError, _splitParam, 'FOO=\\xN')
        self.assertRaises(ValueError, _splitParam, 'FOO=\\x20\\x')

    def test_splitParamArgs(self):
        """
        L{ServerSupportedFeatures._splitParamArgs} splits ISUPPORT parameter
        arguments into key and value.  Arguments without a separator are
        split into a key and an empty string.
        """
        res = irc.ServerSupportedFeatures._splitParamArgs(['A:1', 'B:2', 'C:', 'D'])
        self.assertEqual(res, [('A', '1'), ('B', '2'), ('C', ''), ('D', '')])

    def test_splitParamArgsProcessor(self):
        """
        L{ServerSupportedFeatures._splitParamArgs} uses the argument processor
        passed to convert ISUPPORT argument values to some more suitable
        form.
        """
        res = irc.ServerSupportedFeatures._splitParamArgs(['A:1', 'B:2', 'C'], irc._intOrDefault)
        self.assertEqual(res, [('A', 1), ('B', 2), ('C', None)])

    def test_parsePrefixParam(self):
        """
        L{ServerSupportedFeatures._parsePrefixParam} parses the ISUPPORT PREFIX
        parameter into a mapping from modes to prefix symbols, returns
        L{None} if there is no parseable prefix parameter or raises
        C{ValueError} if the prefix parameter is malformed.
        """
        _parsePrefixParam = irc.ServerSupportedFeatures._parsePrefixParam
        self.assertEqual(_parsePrefixParam(''), None)
        self.assertRaises(ValueError, _parsePrefixParam, 'hello')
        self.assertEqual(_parsePrefixParam('(ov)@+'), {'o': ('@', 0), 'v': ('+', 1)})

    def test_parseChanModesParam(self):
        """
        L{ServerSupportedFeatures._parseChanModesParam} parses the ISUPPORT
        CHANMODES parameter into a mapping from mode categories to mode
        characters. Passing fewer than 4 parameters results in the empty string
        for the relevant categories. Passing more than 4 parameters raises
        C{ValueError}.
        """
        _parseChanModesParam = irc.ServerSupportedFeatures._parseChanModesParam
        self.assertEqual(_parseChanModesParam(['', '', '', '']), {'addressModes': '', 'param': '', 'setParam': '', 'noParam': ''})
        self.assertEqual(_parseChanModesParam(['b', 'k', 'l', 'imnpst']), {'addressModes': 'b', 'param': 'k', 'setParam': 'l', 'noParam': 'imnpst'})
        self.assertEqual(_parseChanModesParam(['b', 'k', 'l', '']), {'addressModes': 'b', 'param': 'k', 'setParam': 'l', 'noParam': ''})
        self.assertRaises(ValueError, _parseChanModesParam, ['a', 'b', 'c', 'd', 'e'])

    def test_parse(self):
        """
        L{ServerSupportedFeatures.parse} changes the internal state of the
        instance to reflect the features indicated by the parsed ISUPPORT
        parameters, including unknown parameters and unsetting previously set
        parameters.
        """
        supported = irc.ServerSupportedFeatures()
        supported.parse(['MODES=4', 'CHANLIMIT=#:20,&:10', 'INVEX', 'EXCEPTS=Z', 'UNKNOWN=A,B,C'])
        self.assertEqual(supported.getFeature('MODES'), 4)
        self.assertEqual(supported.getFeature('CHANLIMIT'), [('#', 20), ('&', 10)])
        self.assertEqual(supported.getFeature('INVEX'), 'I')
        self.assertEqual(supported.getFeature('EXCEPTS'), 'Z')
        self.assertEqual(supported.getFeature('UNKNOWN'), ('A', 'B', 'C'))
        self.assertTrue(supported.hasFeature('INVEX'))
        supported.parse(['-INVEX'])
        self.assertFalse(supported.hasFeature('INVEX'))
        supported.parse(['-INVEX'])

    def _parse(self, features):
        """
        Parse all specified features according to the ISUPPORT specifications.

        @type features: C{list} of C{(featureName, value)}
        @param features: Feature names and values to parse

        @rtype: L{irc.ServerSupportedFeatures}
        """
        supported = irc.ServerSupportedFeatures()
        features = ['{}={}'.format(name, value or '') for name, value in features]
        supported.parse(features)
        return supported

    def _parseFeature(self, name, value=None):
        """
        Parse a feature, with the given name and value, according to the
        ISUPPORT specifications and return the parsed value.
        """
        supported = self._parse([(name, value)])
        return supported.getFeature(name)

    def _testIntOrDefaultFeature(self, name, default=None):
        """
        Perform some common tests on a feature known to use L{_intOrDefault}.
        """
        self.assertEqual(self._parseFeature(name, None), default)
        self.assertEqual(self._parseFeature(name, 'notanint'), default)
        self.assertEqual(self._parseFeature(name, '42'), 42)

    def _testFeatureDefault(self, name, features=None):
        """
        Features known to have default values are reported as being present by
        L{irc.ServerSupportedFeatures.hasFeature}, and their value defaults
        correctly, when they don't appear in an ISUPPORT message.
        """
        default = irc.ServerSupportedFeatures()._features[name]
        if features is None:
            features = [('DEFINITELY_NOT', 'a_feature')]
        supported = self._parse(features)
        self.assertTrue(supported.hasFeature(name))
        self.assertEqual(supported.getFeature(name), default)

    def test_support_CHANMODES(self):
        """
        The CHANMODES ISUPPORT parameter is parsed into a C{dict} giving the
        four mode categories, C{'addressModes'}, C{'param'}, C{'setParam'}, and
        C{'noParam'}.
        """
        self._testFeatureDefault('CHANMODES')
        self._testFeatureDefault('CHANMODES', [('CHANMODES', 'b,,lk,')])
        self._testFeatureDefault('CHANMODES', [('CHANMODES', 'b,,lk,ha,ha')])
        self.assertEqual(self._parseFeature('CHANMODES', ',,,'), {'addressModes': '', 'param': '', 'setParam': '', 'noParam': ''})
        self.assertEqual(self._parseFeature('CHANMODES', ',A,,'), {'addressModes': '', 'param': 'A', 'setParam': '', 'noParam': ''})
        self.assertEqual(self._parseFeature('CHANMODES', 'A,Bc,Def,Ghij'), {'addressModes': 'A', 'param': 'Bc', 'setParam': 'Def', 'noParam': 'Ghij'})

    def test_support_IDCHAN(self):
        """
        The IDCHAN support parameter is parsed into a sequence of two-tuples
        giving channel prefix and ID length pairs.
        """
        self.assertEqual(self._parseFeature('IDCHAN', '!:5'), [('!', '5')])

    def test_support_MAXLIST(self):
        """
        The MAXLIST support parameter is parsed into a sequence of two-tuples
        giving modes and their limits.
        """
        self.assertEqual(self._parseFeature('MAXLIST', 'b:25,eI:50'), [('b', 25), ('eI', 50)])
        self.assertEqual(self._parseFeature('MAXLIST', 'b:25,eI:50,a:3.1415'), [('b', 25), ('eI', 50), ('a', None)])
        self.assertEqual(self._parseFeature('MAXLIST', 'b:25,eI:50,a:notanint'), [('b', 25), ('eI', 50), ('a', None)])

    def test_support_NETWORK(self):
        """
        The NETWORK support parameter is parsed as the network name, as
        specified by the server.
        """
        self.assertEqual(self._parseFeature('NETWORK', 'IRCNet'), 'IRCNet')

    def test_support_SAFELIST(self):
        """
        The SAFELIST support parameter is parsed into a boolean indicating
        whether the safe "list" command is supported or not.
        """
        self.assertEqual(self._parseFeature('SAFELIST'), True)

    def test_support_STATUSMSG(self):
        """
        The STATUSMSG support parameter is parsed into a string of channel
        status that support the exclusive channel notice method.
        """
        self.assertEqual(self._parseFeature('STATUSMSG', '@+'), '@+')

    def test_support_TARGMAX(self):
        """
        The TARGMAX support parameter is parsed into a dictionary, mapping
        strings to integers, of the maximum number of targets for a particular
        command.
        """
        self.assertEqual(self._parseFeature('TARGMAX', 'PRIVMSG:4,NOTICE:3'), {'PRIVMSG': 4, 'NOTICE': 3})
        self.assertEqual(self._parseFeature('TARGMAX', 'PRIVMSG:4,NOTICE:3,KICK:3.1415'), {'PRIVMSG': 4, 'NOTICE': 3, 'KICK': None})
        self.assertEqual(self._parseFeature('TARGMAX', 'PRIVMSG:4,NOTICE:3,KICK:notanint'), {'PRIVMSG': 4, 'NOTICE': 3, 'KICK': None})

    def test_support_NICKLEN(self):
        """
        The NICKLEN support parameter is parsed into an integer value
        indicating the maximum length of a nickname the client may use,
        otherwise, if the parameter is missing or invalid, the default value
        (as specified by RFC 1459) is used.
        """
        default = irc.ServerSupportedFeatures()._features['NICKLEN']
        self._testIntOrDefaultFeature('NICKLEN', default)

    def test_support_CHANNELLEN(self):
        """
        The CHANNELLEN support parameter is parsed into an integer value
        indicating the maximum channel name length, otherwise, if the
        parameter is missing or invalid, the default value (as specified by
        RFC 1459) is used.
        """
        default = irc.ServerSupportedFeatures()._features['CHANNELLEN']
        self._testIntOrDefaultFeature('CHANNELLEN', default)

    def test_support_CHANTYPES(self):
        """
        The CHANTYPES support parameter is parsed into a tuple of
        valid channel prefix characters.
        """
        self._testFeatureDefault('CHANTYPES')
        self.assertEqual(self._parseFeature('CHANTYPES', '#&%'), ('#', '&', '%'))

    def test_support_KICKLEN(self):
        """
        The KICKLEN support parameter is parsed into an integer value
        indicating the maximum length of a kick message a client may use.
        """
        self._testIntOrDefaultFeature('KICKLEN')

    def test_support_PREFIX(self):
        """
        The PREFIX support parameter is parsed into a dictionary mapping
        modes to two-tuples of status symbol and priority.
        """
        self._testFeatureDefault('PREFIX')
        self._testFeatureDefault('PREFIX', [('PREFIX', 'hello')])
        self.assertEqual(self._parseFeature('PREFIX', None), None)
        self.assertEqual(self._parseFeature('PREFIX', '(ohv)@%+'), {'o': ('@', 0), 'h': ('%', 1), 'v': ('+', 2)})
        self.assertEqual(self._parseFeature('PREFIX', '(hov)@%+'), {'o': ('%', 1), 'h': ('@', 0), 'v': ('+', 2)})

    def test_support_TOPICLEN(self):
        """
        The TOPICLEN support parameter is parsed into an integer value
        indicating the maximum length of a topic a client may set.
        """
        self._testIntOrDefaultFeature('TOPICLEN')

    def test_support_MODES(self):
        """
        The MODES support parameter is parsed into an integer value
        indicating the maximum number of "variable" modes (defined as being
        modes from C{addressModes}, C{param} or C{setParam} categories for
        the C{CHANMODES} ISUPPORT parameter) which may by set on a channel
        by a single MODE command from a client.
        """
        self._testIntOrDefaultFeature('MODES')

    def test_support_EXCEPTS(self):
        """
        The EXCEPTS support parameter is parsed into the mode character
        to be used for "ban exception" modes. If no parameter is specified
        then the character C{e} is assumed.
        """
        self.assertEqual(self._parseFeature('EXCEPTS', 'Z'), 'Z')
        self.assertEqual(self._parseFeature('EXCEPTS'), 'e')

    def test_support_INVEX(self):
        """
        The INVEX support parameter is parsed into the mode character to be
        used for "invite exception" modes. If no parameter is specified then
        the character C{I} is assumed.
        """
        self.assertEqual(self._parseFeature('INVEX', 'Z'), 'Z')
        self.assertEqual(self._parseFeature('INVEX'), 'I')