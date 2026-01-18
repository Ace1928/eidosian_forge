from urllib.parse import urlparse
from dulwich.tests import TestCase
from ..config import ConfigDict
from ..credentials import match_partial_url, match_urls, urlmatch_credential_sections
class TestCredentialHelpersUtils(TestCase):

    def test_match_urls(self):
        url = urlparse('https://github.com/jelmer/dulwich/')
        url_1 = urlparse('https://github.com/jelmer/dulwich')
        url_2 = urlparse('https://github.com/jelmer')
        url_3 = urlparse('https://github.com')
        self.assertTrue(match_urls(url, url_1))
        self.assertTrue(match_urls(url, url_2))
        self.assertTrue(match_urls(url, url_3))
        non_matching = urlparse('https://git.sr.ht/')
        self.assertFalse(match_urls(url, non_matching))

    def test_match_partial_url(self):
        url = urlparse('https://github.com/jelmer/dulwich/')
        self.assertTrue(match_partial_url(url, 'github.com'))
        self.assertFalse(match_partial_url(url, 'github.com/jelmer/'))
        self.assertTrue(match_partial_url(url, 'github.com/jelmer/dulwich'))
        self.assertFalse(match_partial_url(url, 'github.com/jel'))
        self.assertFalse(match_partial_url(url, 'github.com/jel/'))

    def test_urlmatch_credential_sections(self):
        config = ConfigDict()
        config.set((b'credential', 'https://github.com'), b'helper', 'foo')
        config.set((b'credential', 'git.sr.ht'), b'helper', 'foo')
        config.set(b'credential', b'helper', 'bar')
        self.assertEqual(list(urlmatch_credential_sections(config, 'https://github.com')), [(b'credential', b'https://github.com'), (b'credential',)])
        self.assertEqual(list(urlmatch_credential_sections(config, 'https://git.sr.ht')), [(b'credential', b'git.sr.ht'), (b'credential',)])
        self.assertEqual(list(urlmatch_credential_sections(config, 'missing_url')), [(b'credential',)])