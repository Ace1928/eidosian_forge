from collections import defaultdict
import unittest
from lazr.uri import (
class URITestCase(unittest.TestCase):

    def test_normalisation(self):
        self.assertEqual(str(URI('eXAMPLE://a/./b/../b/%63/%7bfoo%7d')), 'example://a/b/c/%7Bfoo%7D')
        self.assertEqual(str(URI('http://www.EXAMPLE.com/')), 'http://www.example.com/')
        self.assertEqual(str(URI('http://www.gnome.org/%7ejamesh/')), 'http://www.gnome.org/~jamesh/')
        self.assertEqual(str(URI('http://example.com')), 'http://example.com/')
        self.assertEqual(str(URI('http://example.com:/')), 'http://example.com/')
        self.assertEqual(str(URI('http://example.com:80/')), 'http://example.com/')

    def test_hashable(self):
        uri_groups = [['eXAMPLE://a/./b/../b/%63/%7bfoo%7d', 'example://a/b/c/%7Bfoo%7D'], ['http://www.EXAMPLE.com/', 'http://www.example.com/'], ['http://www.gnome.org/%7ejamesh/', 'http://www.gnome.org/~jamesh/'], ['http://example.com', 'http://example.com/', 'http://example.com:/', 'http://example.com:80/']]
        uri_hashes = defaultdict(list)
        for uri_group in uri_groups:
            for uri in uri_group:
                uri_hashes[hash(URI(uri))].append(uri)
        self.assertEqual(len(uri_groups), len(uri_hashes))
        for uri_group in uri_groups:
            self.assertEqual(sorted(uri_group), sorted(uri_hashes[hash(URI(uri_group[0]))]))

    def test_invalid_uri(self):
        self.assertRaises(InvalidURIError, URI, 'http://€xample.com/')

    def test_merge(self):
        self.assertEqual(merge('', 'foo', has_authority=True), '/foo')
        self.assertEqual(merge('', 'foo', has_authority=False), 'foo')
        self.assertEqual(merge('/a/b/c', 'foo', has_authority=True), '/a/b/foo')
        self.assertEqual(merge('/a/b/', 'foo', has_authority=True), '/a/b/foo')

    def test_remove_dot_segments(self):
        self.assertEqual(remove_dot_segments('/a/b/c/./../../g'), '/a/g')
        self.assertEqual(remove_dot_segments('mid/content=5/../6'), 'mid/6')

    def test_normal_resolution(self):
        base = URI('http://a/b/c/d;p?q')

        def resolve(relative):
            return str(base.resolve(relative))
        self.assertEqual(resolve('g:h'), 'g:h')
        self.assertEqual(resolve('g'), 'http://a/b/c/g')
        self.assertEqual(resolve('./g'), 'http://a/b/c/g')
        self.assertEqual(resolve('g/'), 'http://a/b/c/g/')
        self.assertEqual(resolve('/g'), 'http://a/g')
        self.assertEqual(resolve('//g'), 'http://g/')
        self.assertEqual(resolve('?y'), 'http://a/b/c/d;p?y')
        self.assertEqual(resolve('g?y'), 'http://a/b/c/g?y')
        self.assertEqual(resolve('#s'), 'http://a/b/c/d;p?q#s')
        self.assertEqual(resolve('g#s'), 'http://a/b/c/g#s')
        self.assertEqual(resolve('g?y#s'), 'http://a/b/c/g?y#s')
        self.assertEqual(resolve(';x'), 'http://a/b/c/;x')
        self.assertEqual(resolve('g;x'), 'http://a/b/c/g;x')
        self.assertEqual(resolve('g;x?y#s'), 'http://a/b/c/g;x?y#s')
        self.assertEqual(resolve(''), 'http://a/b/c/d;p?q')
        self.assertEqual(resolve('.'), 'http://a/b/c/')
        self.assertEqual(resolve('./'), 'http://a/b/c/')
        self.assertEqual(resolve('..'), 'http://a/b/')
        self.assertEqual(resolve('../'), 'http://a/b/')
        self.assertEqual(resolve('../g'), 'http://a/b/g')
        self.assertEqual(resolve('../..'), 'http://a/')
        self.assertEqual(resolve('../../'), 'http://a/')
        self.assertEqual(resolve('../../g'), 'http://a/g')

    def test_abnormal_resolution(self):
        base = URI('http://a/b/c/d;p?q')

        def resolve(relative):
            return str(base.resolve(relative))
        self.assertEqual(resolve('../../../g'), 'http://a/g')
        self.assertEqual(resolve('../../../../g'), 'http://a/g')
        self.assertEqual(resolve('/./g'), 'http://a/g')
        self.assertEqual(resolve('/../g'), 'http://a/g')
        self.assertEqual(resolve('g.'), 'http://a/b/c/g.')
        self.assertEqual(resolve('.g'), 'http://a/b/c/.g')
        self.assertEqual(resolve('g..'), 'http://a/b/c/g..')
        self.assertEqual(resolve('..g'), 'http://a/b/c/..g')
        self.assertEqual(resolve('./../g'), 'http://a/b/g')
        self.assertEqual(resolve('./g/.'), 'http://a/b/c/g/')
        self.assertEqual(resolve('g/./h'), 'http://a/b/c/g/h')
        self.assertEqual(resolve('g/../h'), 'http://a/b/c/h')
        self.assertEqual(resolve('g;x=1/./y'), 'http://a/b/c/g;x=1/y')
        self.assertEqual(resolve('g;x=1/../y'), 'http://a/b/c/y')
        self.assertEqual(resolve('g?y/./x'), 'http://a/b/c/g?y/./x')
        self.assertEqual(resolve('g?y/../x'), 'http://a/b/c/g?y/../x')
        self.assertEqual(resolve('g#s/./x'), 'http://a/b/c/g#s/./x')
        self.assertEqual(resolve('g#s/../x'), 'http://a/b/c/g#s/../x')

    def test_underDomain_matches_subdomain(self):
        uri = URI('http://code.launchpad.dev/foo')
        self.assertTrue(uri.underDomain('code.launchpad.dev'))
        self.assertTrue(uri.underDomain('launchpad.dev'))
        self.assertTrue(uri.underDomain(''))

    def test_underDomain_doesnt_match_non_subdomain(self):
        uri = URI('http://code.launchpad.dev/foo')
        self.assertFalse(uri.underDomain('beta.code.launchpad.dev'))
        self.assertFalse(uri.underDomain('google.com'))
        self.assertFalse(uri.underDomain('unchpad.dev'))