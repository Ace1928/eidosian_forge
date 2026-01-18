from ... import tests
from .. import generate_ids
class TestGenRevisionId(tests.TestCase):
    """Test generating revision ids"""

    def assertGenRevisionId(self, regex, username, timestamp=None):
        """gen_revision_id should create a revision id matching the regex"""
        revision_id = generate_ids.gen_revision_id(username, timestamp)
        self.assertContainsRe(revision_id, b'^' + regex + b'$')
        self.assertIsInstance(revision_id, bytes)
        revision_id.decode('ascii')

    def test_timestamp(self):
        """passing a timestamp should cause it to be used"""
        self.assertGenRevisionId(b'user@host-\\d{14}-[a-z0-9]{16}', 'user@host')
        self.assertGenRevisionId(b'user@host-20061102205056-[a-z0-9]{16}', 'user@host', 1162500656.688)
        self.assertGenRevisionId(b'user@host-20061102205024-[a-z0-9]{16}', 'user@host', 1162500624.0)

    def test_gen_revision_id_email(self):
        """gen_revision_id uses email address if present"""
        regex = b'user\\+joe_bar@foo-bar\\.com-\\d{14}-[a-z0-9]{16}'
        self.assertGenRevisionId(regex, 'user+joe_bar@foo-bar.com')
        self.assertGenRevisionId(regex, '<user+joe_bar@foo-bar.com>')
        self.assertGenRevisionId(regex, 'Joe Bar <user+joe_bar@foo-bar.com>')
        self.assertGenRevisionId(regex, 'Joe Bar <user+Joe_Bar@Foo-Bar.com>')
        self.assertGenRevisionId(regex, 'Joe Bår <user+Joe_Bar@Foo-Bar.com>')

    def test_gen_revision_id_user(self):
        """If there is no email, fall back to the whole username"""
        tail = b'-\\d{14}-[a-z0-9]{16}'
        self.assertGenRevisionId(b'joe_bar' + tail, 'Joe Bar')
        self.assertGenRevisionId(b'joebar' + tail, 'joebar')
        self.assertGenRevisionId(b'joe_br' + tail, 'Joe Bår')
        self.assertGenRevisionId(b'joe_br_user\\+joe_bar_foo-bar.com' + tail, 'Joe Bår <user+Joe_Bar_Foo-Bar.com>')

    def test_revision_ids_are_ascii(self):
        """gen_revision_id should always return an ascii revision id."""
        tail = b'-\\d{14}-[a-z0-9]{16}'
        self.assertGenRevisionId(b'joe_bar' + tail, 'Joe Bar')
        self.assertGenRevisionId(b'joe_bar' + tail, 'Joe Bar')
        self.assertGenRevisionId(b'joe@foo' + tail, 'Joe Bar <joe@foo>')
        self.assertGenRevisionId(b'joe@f' + tail, 'Joe Bar <joe@f¶>')