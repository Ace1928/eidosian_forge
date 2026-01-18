import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
class ServerSupportedFeatures(_CommandDispatcherMixin):
    """
    Handle ISUPPORT messages.

    Feature names match those in the ISUPPORT RFC draft identically.

    Information regarding the specifics of ISUPPORT was gleaned from
    <http://www.irc.org/tech_docs/draft-brocklesby-irc-isupport-03.txt>.
    """
    prefix = 'isupport'

    def __init__(self):
        self._features = {'CHANNELLEN': 200, 'CHANTYPES': tuple('#&'), 'MODES': 3, 'NICKLEN': 9, 'PREFIX': self._parsePrefixParam('(ovh)@+%'), 'CHANMODES': self._parseChanModesParam(['b', '', 'lk', ''])}

    @classmethod
    def _splitParamArgs(cls, params, valueProcessor=None):
        """
        Split ISUPPORT parameter arguments.

        Values can optionally be processed by C{valueProcessor}.

        For example::

            >>> ServerSupportedFeatures._splitParamArgs(['A:1', 'B:2'])
            (('A', '1'), ('B', '2'))

        @type params: C{iterable} of C{str}

        @type valueProcessor: C{callable} taking {str}
        @param valueProcessor: Callable to process argument values, or L{None}
            to perform no processing

        @rtype: C{list} of C{(str, object)}
        @return: Sequence of C{(name, processedValue)}
        """
        if valueProcessor is None:
            valueProcessor = lambda x: x

        def _parse():
            for param in params:
                if ':' not in param:
                    param += ':'
                a, b = param.split(':', 1)
                yield (a, valueProcessor(b))
        return list(_parse())

    @classmethod
    def _unescapeParamValue(cls, value):
        """
        Unescape an ISUPPORT parameter.

        The only form of supported escape is C{\\xHH}, where HH must be a valid
        2-digit hexadecimal number.

        @rtype: C{str}
        """

        def _unescape():
            parts = value.split('\\x')
            yield parts.pop(0)
            for s in parts:
                octet, rest = (s[:2], s[2:])
                try:
                    octet = int(octet, 16)
                except ValueError:
                    raise ValueError(f'Invalid hex octet: {octet!r}')
                yield (chr(octet) + rest)
        if '\\x' not in value:
            return value
        return ''.join(_unescape())

    @classmethod
    def _splitParam(cls, param):
        """
        Split an ISUPPORT parameter.

        @type param: C{str}

        @rtype: C{(str, list)}
        @return: C{(key, arguments)}
        """
        if '=' not in param:
            param += '='
        key, value = param.split('=', 1)
        return (key, [cls._unescapeParamValue(v) for v in value.split(',')])

    @classmethod
    def _parsePrefixParam(cls, prefix):
        """
        Parse the ISUPPORT "PREFIX" parameter.

        The order in which the parameter arguments appear is significant, the
        earlier a mode appears the more privileges it gives.

        @rtype: C{dict} mapping C{str} to C{(str, int)}
        @return: A dictionary mapping a mode character to a two-tuple of
            C({symbol, priority)}, the lower a priority (the lowest being
            C{0}) the more privileges it gives
        """
        if not prefix:
            return None
        if prefix[0] != '(' and ')' not in prefix:
            raise ValueError('Malformed PREFIX parameter')
        modes, symbols = prefix.split(')', 1)
        symbols = zip(symbols, range(len(symbols)))
        modes = modes[1:]
        return dict(zip(modes, symbols))

    @classmethod
    def _parseChanModesParam(self, params):
        """
        Parse the ISUPPORT "CHANMODES" parameter.

        See L{isupport_CHANMODES} for a detailed explanation of this parameter.
        """
        names = ('addressModes', 'param', 'setParam', 'noParam')
        if len(params) > len(names):
            raise ValueError('Expecting a maximum of %d channel mode parameters, got %d' % (len(names), len(params)))
        items = map(lambda key, value: (key, value or ''), names, params)
        return dict(items)

    def getFeature(self, feature, default=None):
        """
        Get a server supported feature's value.

        A feature with the value L{None} is equivalent to the feature being
        unsupported.

        @type feature: C{str}
        @param feature: Feature name

        @type default: C{object}
        @param default: The value to default to, assuming that C{feature}
            is not supported

        @return: Feature value
        """
        return self._features.get(feature, default)

    def hasFeature(self, feature):
        """
        Determine whether a feature is supported or not.

        @rtype: C{bool}
        """
        return self.getFeature(feature) is not None

    def parse(self, params):
        """
        Parse ISUPPORT parameters.

        If an unknown parameter is encountered, it is simply added to the
        dictionary, keyed by its name, as a tuple of the parameters provided.

        @type params: C{iterable} of C{str}
        @param params: Iterable of ISUPPORT parameters to parse
        """
        for param in params:
            key, value = self._splitParam(param)
            if key.startswith('-'):
                self._features.pop(key[1:], None)
            else:
                self._features[key] = self.dispatch(key, value)

    def isupport_unknown(self, command, params):
        """
        Unknown ISUPPORT parameter.
        """
        return tuple(params)

    def isupport_CHANLIMIT(self, params):
        """
        The maximum number of each channel type a user may join.
        """
        return self._splitParamArgs(params, _intOrDefault)

    def isupport_CHANMODES(self, params):
        """
        Available channel modes.

        There are 4 categories of channel mode::

            addressModes - Modes that add or remove an address to or from a
            list, these modes always take a parameter.

            param - Modes that change a setting on a channel, these modes
            always take a parameter.

            setParam - Modes that change a setting on a channel, these modes
            only take a parameter when being set.

            noParam - Modes that change a setting on a channel, these modes
            never take a parameter.
        """
        try:
            return self._parseChanModesParam(params)
        except ValueError:
            return self.getFeature('CHANMODES')

    def isupport_CHANNELLEN(self, params):
        """
        Maximum length of a channel name a client may create.
        """
        return _intOrDefault(params[0], self.getFeature('CHANNELLEN'))

    def isupport_CHANTYPES(self, params):
        """
        Valid channel prefixes.
        """
        return tuple(params[0])

    def isupport_EXCEPTS(self, params):
        """
        Mode character for "ban exceptions".

        The presence of this parameter indicates that the server supports
        this functionality.
        """
        return params[0] or 'e'

    def isupport_IDCHAN(self, params):
        """
        Safe channel identifiers.

        The presence of this parameter indicates that the server supports
        this functionality.
        """
        return self._splitParamArgs(params)

    def isupport_INVEX(self, params):
        """
        Mode character for "invite exceptions".

        The presence of this parameter indicates that the server supports
        this functionality.
        """
        return params[0] or 'I'

    def isupport_KICKLEN(self, params):
        """
        Maximum length of a kick message a client may provide.
        """
        return _intOrDefault(params[0])

    def isupport_MAXLIST(self, params):
        """
        Maximum number of "list modes" a client may set on a channel at once.

        List modes are identified by the "addressModes" key in CHANMODES.
        """
        return self._splitParamArgs(params, _intOrDefault)

    def isupport_MODES(self, params):
        """
        Maximum number of modes accepting parameters that may be sent, by a
        client, in a single MODE command.
        """
        return _intOrDefault(params[0])

    def isupport_NETWORK(self, params):
        """
        IRC network name.
        """
        return params[0]

    def isupport_NICKLEN(self, params):
        """
        Maximum length of a nickname the client may use.
        """
        return _intOrDefault(params[0], self.getFeature('NICKLEN'))

    def isupport_PREFIX(self, params):
        """
        Mapping of channel modes that clients may have to status flags.
        """
        try:
            return self._parsePrefixParam(params[0])
        except ValueError:
            return self.getFeature('PREFIX')

    def isupport_SAFELIST(self, params):
        """
        Flag indicating that a client may request a LIST without being
        disconnected due to the large amount of data generated.
        """
        return True

    def isupport_STATUSMSG(self, params):
        """
        The server supports sending messages to only to clients on a channel
        with a specific status.
        """
        return params[0]

    def isupport_TARGMAX(self, params):
        """
        Maximum number of targets allowable for commands that accept multiple
        targets.
        """
        return dict(self._splitParamArgs(params, _intOrDefault))

    def isupport_TOPICLEN(self, params):
        """
        Maximum length of a topic that may be set.
        """
        return _intOrDefault(params[0])