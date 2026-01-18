from typing import Dict, List, Tuple
from twisted.internet.testing import StringTransport
from twisted.protocols import postfix
from twisted.trial import unittest
class ValidTests(PostfixTCPMapServerTestCase, unittest.TestCase):
    data = {b'foo': b'ThisIs Foo', b'bar': b' bar really is found\r\n'}
    chat = [(b'get', b"400 Command 'get' takes 1 parameters.\n"), (b'get foo bar', b'500 \n'), (b'put', b"400 Command 'put' takes 2 parameters.\n"), (b'put foo', b"400 Command 'put' takes 2 parameters.\n"), (b'put foo bar baz', b'500 put is not implemented yet.\n'), (b'put foo bar', b'500 put is not implemented yet.\n'), (b'get foo', b'200 ThisIs%20Foo\n'), (b'get bar', b'200 %20bar%20really%20is%20found%0D%0A\n'), (b'get baz', b'500 \n'), (b'foo', b'400 unknown command\n')]