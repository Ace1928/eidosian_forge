from mock import patch, Mock
import unittest
import time
from boto.exception import S3ResponseError
from boto.s3.connection import S3Connection
from boto.s3.bucketlogging import BucketLogging
from boto.s3.lifecycle import Lifecycle
from boto.s3.lifecycle import Transition
from boto.s3.lifecycle import Expiration
from boto.s3.lifecycle import Rule
from boto.s3.acl import Grant
from boto.s3.tagging import Tags, TagSet
from boto.s3.website import RedirectLocation
from boto.compat import unquote_str
class S3BucketTest(unittest.TestCase):
    s3 = True

    def setUp(self):
        self.conn = S3Connection()
        self.bucket_name = 'bucket-%d' % int(time.time())
        self.bucket = self.conn.create_bucket(self.bucket_name)

    def tearDown(self):
        for key in self.bucket:
            key.delete()
        self.bucket.delete()

    def test_next_marker(self):
        expected = ['a/', 'b', 'c']
        for key_name in expected:
            key = self.bucket.new_key(key_name)
            key.set_contents_from_string(key_name)
        rs = self.bucket.get_all_keys(max_keys=2)
        for element in rs:
            pass
        self.assertEqual(element.name, 'b')
        self.assertEqual(rs.next_marker, None)
        rs = self.bucket.get_all_keys(max_keys=2, delimiter='/')
        for element in rs:
            pass
        self.assertEqual(element.name, 'a/')
        self.assertEqual(rs.next_marker, 'b')
        rs = self.bucket.list()
        for element in rs:
            self.assertEqual(element.name, expected.pop(0))
        self.assertEqual(expected, [])

    def test_list_with_url_encoding(self):
        expected = [u'α', u'β', u'γ']
        for key_name in expected:
            key = self.bucket.new_key(key_name)
            key.set_contents_from_string(key_name)
        orig_getall = self.bucket._get_all
        getall = lambda *a, **k: orig_getall(*a, max_keys=2, **k)
        with patch.object(self.bucket, '_get_all', getall):
            rs = self.bucket.list(encoding_type='url')
            for element in rs:
                name = unquote_str(element.name)
                self.assertEqual(name, expected.pop(0))
            self.assertEqual(expected, [])

    def test_logging(self):
        sb_name = 'src-' + self.bucket_name
        sb = self.conn.create_bucket(sb_name)
        self.bucket.set_acl('log-delivery-write')
        target_bucket = self.bucket_name
        target_prefix = u'jp/ログ/'
        bls = sb.get_logging_status()
        self.assertEqual(bls.target, None)
        authuri = 'http://acs.amazonaws.com/groups/global/AuthenticatedUsers'
        authr = Grant(permission='READ', type='Group', uri=authuri)
        sb.enable_logging(target_bucket, target_prefix=target_prefix, grants=[authr])
        bls = sb.get_logging_status()
        self.assertEqual(bls.target, target_bucket)
        self.assertEqual(bls.prefix, target_prefix)
        self.assertEqual(len(bls.grants), 1)
        self.assertEqual(bls.grants[0].type, 'Group')
        self.assertEqual(bls.grants[0].uri, authuri)
        sb.delete()

    def test_tagging(self):
        tagging = '\n            <Tagging>\n              <TagSet>\n                 <Tag>\n                   <Key>tagkey</Key>\n                   <Value>tagvalue</Value>\n                 </Tag>\n              </TagSet>\n            </Tagging>\n        '
        self.bucket.set_xml_tags(tagging)
        response = self.bucket.get_tags()
        self.assertEqual(response[0][0].key, 'tagkey')
        self.assertEqual(response[0][0].value, 'tagvalue')
        self.bucket.delete_tags()
        try:
            self.bucket.get_tags()
        except S3ResponseError as e:
            self.assertEqual(e.code, 'NoSuchTagSet')
        except Exception as e:
            self.fail('Wrong exception raised (expected S3ResponseError): %s' % e)
        else:
            self.fail('Expected S3ResponseError, but no exception raised.')

    def test_tagging_from_objects(self):
        """Create tags from python objects rather than raw xml."""
        t = Tags()
        tag_set = TagSet()
        tag_set.add_tag('akey', 'avalue')
        tag_set.add_tag('anotherkey', 'anothervalue')
        t.add_tag_set(tag_set)
        self.bucket.set_tags(t)
        response = self.bucket.get_tags()
        tags = sorted(response[0], key=lambda tag: tag.key)
        self.assertEqual(tags[0].key, 'akey')
        self.assertEqual(tags[0].value, 'avalue')
        self.assertEqual(tags[1].key, 'anotherkey')
        self.assertEqual(tags[1].value, 'anothervalue')

    def test_website_configuration(self):
        response = self.bucket.configure_website('index.html')
        self.assertTrue(response)
        config = self.bucket.get_website_configuration()
        self.assertEqual(config, {'WebsiteConfiguration': {'IndexDocument': {'Suffix': 'index.html'}}})
        config2, xml = self.bucket.get_website_configuration_with_xml()
        self.assertEqual(config, config2)
        self.assertTrue('<Suffix>index.html</Suffix>' in xml, xml)

    def test_website_redirect_all_requests(self):
        response = self.bucket.configure_website(redirect_all_requests_to=RedirectLocation('example.com'))
        config = self.bucket.get_website_configuration()
        self.assertEqual(config, {'WebsiteConfiguration': {'RedirectAllRequestsTo': {'HostName': 'example.com'}}})
        response = self.bucket.configure_website(redirect_all_requests_to=RedirectLocation('example.com', 'https'))
        config = self.bucket.get_website_configuration()
        self.assertEqual(config, {'WebsiteConfiguration': {'RedirectAllRequestsTo': {'HostName': 'example.com', 'Protocol': 'https'}}})

    def test_lifecycle(self):
        lifecycle = Lifecycle()
        lifecycle.add_rule('myid', '', 'Enabled', 30)
        self.assertTrue(self.bucket.configure_lifecycle(lifecycle))
        response = self.bucket.get_lifecycle_config()
        self.assertEqual(len(response), 1)
        actual_lifecycle = response[0]
        self.assertEqual(actual_lifecycle.id, 'myid')
        self.assertEqual(actual_lifecycle.prefix, '')
        self.assertEqual(actual_lifecycle.status, 'Enabled')
        self.assertEqual(actual_lifecycle.transition, [])

    def test_lifecycle_with_glacier_transition(self):
        lifecycle = Lifecycle()
        transition = Transition(days=30, storage_class='GLACIER')
        rule = Rule('myid', prefix='', status='Enabled', expiration=None, transition=transition)
        lifecycle.append(rule)
        self.assertTrue(self.bucket.configure_lifecycle(lifecycle))
        response = self.bucket.get_lifecycle_config()
        transition = response[0].transition
        self.assertEqual(transition.days, 30)
        self.assertEqual(transition.storage_class, 'GLACIER')
        self.assertEqual(transition.date, None)

    def test_lifecycle_multi(self):
        date = '2022-10-12T00:00:00.000Z'
        sc = 'GLACIER'
        lifecycle = Lifecycle()
        lifecycle.add_rule('1', '1/', 'Enabled', 1)
        lifecycle.add_rule('2', '2/', 'Enabled', Expiration(days=2))
        lifecycle.add_rule('3', '3/', 'Enabled', Expiration(date=date))
        lifecycle.add_rule('4', '4/', 'Enabled', None, Transition(days=4, storage_class=sc))
        lifecycle.add_rule('5', '5/', 'Enabled', None, Transition(date=date, storage_class=sc))
        self.bucket.configure_lifecycle(lifecycle)
        readlifecycle = self.bucket.get_lifecycle_config()
        for rule in readlifecycle:
            if rule.id == '1':
                self.assertEqual(rule.prefix, '1/')
                self.assertEqual(rule.expiration.days, 1)
            elif rule.id == '2':
                self.assertEqual(rule.prefix, '2/')
                self.assertEqual(rule.expiration.days, 2)
            elif rule.id == '3':
                self.assertEqual(rule.prefix, '3/')
                self.assertEqual(rule.expiration.date, date)
            elif rule.id == '4':
                self.assertEqual(rule.prefix, '4/')
                self.assertEqual(rule.transition.days, 4)
                self.assertEqual(rule.transition.storage_class, sc)
            elif rule.id == '5':
                self.assertEqual(rule.prefix, '5/')
                self.assertEqual(rule.transition.date, date)
                self.assertEqual(rule.transition.storage_class, sc)
            else:
                self.fail('unexpected id %s' % rule.id)

    def test_lifecycle_jp(self):
        name = 'Japanese files'
        prefix = '日本語/'
        days = 30
        lifecycle = Lifecycle()
        lifecycle.add_rule(name, prefix, 'Enabled', days)
        self.bucket.configure_lifecycle(lifecycle)
        readlifecycle = self.bucket.get_lifecycle_config()
        for rule in readlifecycle:
            self.assertEqual(rule.id, name)
            self.assertEqual(rule.expiration.days, days)

    def test_lifecycle_with_defaults(self):
        lifecycle = Lifecycle()
        lifecycle.add_rule(expiration=30)
        self.assertTrue(self.bucket.configure_lifecycle(lifecycle))
        response = self.bucket.get_lifecycle_config()
        self.assertEqual(len(response), 1)
        actual_lifecycle = response[0]
        self.assertNotEqual(len(actual_lifecycle.id), 0)
        self.assertEqual(actual_lifecycle.prefix, '')

    def test_lifecycle_rule_xml(self):
        rule = Rule(status='Enabled', expiration=30)
        s = rule.to_xml()
        self.assertEqual(s.find('<ID>'), -1)
        self.assertNotEqual(s.find('<Prefix></Prefix>'), -1)