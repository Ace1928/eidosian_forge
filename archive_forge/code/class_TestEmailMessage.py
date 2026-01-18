import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
class TestEmailMessage(tests.TestCase):

    def test_empty_message(self):
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
        self.assertEqualDiff(EMPTY_MESSAGE, msg.as_string())

    def test_simple_message(self):
        pairs = {b'body': SIMPLE_MESSAGE_ASCII, 'bódy': SIMPLE_MESSAGE_UTF8, b'b\xc3\xb3dy': SIMPLE_MESSAGE_UTF8, b'b\xf4dy': SIMPLE_MESSAGE_8BIT}
        for body, expected in pairs.items():
            msg = EmailMessage('from@from.com', 'to@to.com', 'subject', body)
            self.assertEqualDiff(expected, msg.as_string())

    def test_multipart_message_simple(self):
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
        msg.add_inline_attachment('body')
        self.assertEqualDiff(simple_multipart_message(), msg.as_string(BOUNDARY))

    def test_multipart_message_complex(self):
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject', 'body')
        msg.add_inline_attachment('a\nb\nc\nd\ne\n', 'lines.txt', 'x-subtype')
        self.assertEqualDiff(complex_multipart_message('x-subtype'), msg.as_string(BOUNDARY))

    def test_headers_accept_unicode_and_utf8(self):
        for user in ['Pepe Pérez <pperez@ejemplo.com>', 'Pepe PÃ©red <pperez@ejemplo.com>']:
            msg = EmailMessage(user, user, user)
            for header in ['From', 'To', 'Subject']:
                value = msg[header]
                value.encode('ascii')

    def test_headers_reject_8bit(self):
        for i in range(3):
            x = [b'"J. Random Developer" <jrandom@example.com>'] * 3
            x[i] = b'Pepe P\xe9rez <pperez@ejemplo.com>'
            self.assertRaises(BzrBadParameterNotUnicode, EmailMessage, *x)

    def test_multiple_destinations(self):
        to_addresses = ['to1@to.com', 'to2@to.com', 'to3@to.com']
        msg = EmailMessage('from@from.com', to_addresses, 'subject')
        self.assertContainsRe(msg.as_string(), 'To: ' + ', '.join(to_addresses))

    def test_retrieving_headers(self):
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
        for header, value in [('From', 'from@from.com'), ('To', 'to@to.com'), ('Subject', 'subject')]:
            self.assertEqual(value, msg.get(header))
            self.assertEqual(value, msg[header])
        self.assertEqual(None, msg.get('Does-Not-Exist'))
        self.assertEqual(None, msg['Does-Not-Exist'])
        self.assertEqual('None', msg.get('Does-Not-Exist', 'None'))

    def test_setting_headers(self):
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
        msg['To'] = 'to2@to.com'
        msg['Cc'] = 'cc@cc.com'
        self.assertEqual('to2@to.com', msg['To'])
        self.assertEqual('cc@cc.com', msg['Cc'])

    def test_address_to_encoded_header(self):

        def decode(s):
            """Convert a RFC2047-encoded string to a unicode string."""
            return ''.join([chunk.decode(encoding or 'ascii') for chunk, encoding in decode_header(s)])
        address = 'jrandom@example.com'
        encoded = EmailMessage.address_to_encoded_header(address)
        self.assertEqual(address, encoded)
        address = 'J Random Developer <jrandom@example.com>'
        encoded = EmailMessage.address_to_encoded_header(address)
        self.assertEqual(address, encoded)
        address = '"J. Random Developer" <jrandom@example.com>'
        encoded = EmailMessage.address_to_encoded_header(address)
        self.assertEqual(address, encoded)
        address = 'Pepe Pérez <pperez@ejemplo.com>'
        encoded = EmailMessage.address_to_encoded_header(address)
        self.assertTrue('pperez@ejemplo.com' in encoded)
        self.assertEqual(address, decode(encoded))
        address = b'Pepe P\xe9rez <pperez@ejemplo.com>'
        self.assertRaises(BzrBadParameterNotUnicode, EmailMessage.address_to_encoded_header, address)

    def test_string_with_encoding(self):
        pairs = {'Pepe': (b'Pepe', 'ascii'), 'Pérez': (b'P\xc3\xa9rez', 'utf-8'), b'P\xc3\xa9rez': (b'P\xc3\xa9rez', 'utf-8'), b'P\xe8rez': (b'P\xe8rez', '8-bit')}
        for string_, pair in pairs.items():
            self.assertEqual(pair, EmailMessage.string_with_encoding(string_))