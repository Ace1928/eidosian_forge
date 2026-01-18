import os
import re
import urllib
import xml.sax
from six import StringIO
from boto import handler
from boto import storage_uri
from boto.gs.acl import ACL
from boto.gs.cors import Cors
from boto.gs.lifecycle import LifecycleConfig
from tests.integration.gs.testcase import GSTestCase
class GSBasicTest(GSTestCase):
    """Tests some basic GCS functionality."""

    def test_read_write(self):
        """Tests basic read/write to keys."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        bucket = self._GetConnection().get_bucket(bucket_name)
        key_name = 'foobar'
        k = bucket.new_key(key_name)
        s1 = 'This is a test of file upload and download'
        k.set_contents_from_string(s1)
        tmpdir = self._MakeTempDir()
        fpath = os.path.join(tmpdir, key_name)
        fp = open(fpath, 'wb')
        k.get_contents_to_file(fp)
        fp.close()
        fp = open(fpath)
        self.assertEqual(s1, fp.read())
        fp.close()
        url = self._conn.generate_url(900, 'GET', bucket=bucket.name, key=key_name)
        f = urllib.urlopen(url)
        self.assertEqual(s1, f.read())
        f.close()
        sfp = StringIO.StringIO('foo')
        k.set_contents_from_file(sfp)
        self.assertEqual(k.get_contents_as_string(), 'foo')
        sfp2 = StringIO.StringIO('foo2')
        k.set_contents_from_file(sfp2)
        self.assertEqual(k.get_contents_as_string(), 'foo2')

    def test_get_all_keys(self):
        """Tests get_all_keys."""
        phony_mimetype = 'application/x-boto-test'
        headers = {'Content-Type': phony_mimetype}
        tmpdir = self._MakeTempDir()
        fpath = os.path.join(tmpdir, 'foobar1')
        fpath2 = os.path.join(tmpdir, 'foobar')
        with open(fpath2, 'w') as f:
            f.write('test-data')
        bucket = self._MakeBucket()
        k = bucket.new_key('foobar')
        s1 = 'test-contents'
        s2 = 'test-contents2'
        k.name = 'foo/bar'
        k.set_contents_from_string(s1, headers)
        k.name = 'foo/bas'
        k.set_contents_from_filename(fpath2)
        k.name = 'foo/bat'
        k.set_contents_from_string(s1)
        k.name = 'fie/bar'
        k.set_contents_from_string(s1)
        k.name = 'fie/bas'
        k.set_contents_from_string(s1)
        k.name = 'fie/bat'
        k.set_contents_from_string(s1)
        md5 = k.md5
        k.set_contents_from_string(s2)
        self.assertNotEqual(k.md5, md5)
        fp2 = open(fpath2, 'rb')
        k.md5 = None
        k.base64md5 = None
        k.set_contents_from_stream(fp2)
        fp = open(fpath, 'wb')
        k.get_contents_to_file(fp)
        fp.close()
        fp2.seek(0, 0)
        fp = open(fpath, 'rb')
        self.assertEqual(fp2.read(), fp.read())
        fp.close()
        fp2.close()
        all = bucket.get_all_keys()
        self.assertEqual(len(all), 6)
        rs = bucket.get_all_keys(prefix='foo')
        self.assertEqual(len(rs), 3)
        rs = bucket.get_all_keys(prefix='', delimiter='/')
        self.assertEqual(len(rs), 2)
        rs = bucket.get_all_keys(maxkeys=5)
        self.assertEqual(len(rs), 5)

    def test_bucket_lookup(self):
        """Test the bucket lookup method."""
        bucket = self._MakeBucket()
        k = bucket.new_key('foo/bar')
        phony_mimetype = 'application/x-boto-test'
        headers = {'Content-Type': phony_mimetype}
        k.set_contents_from_string('testdata', headers)
        k = bucket.lookup('foo/bar')
        self.assertIsInstance(k, bucket.key_class)
        self.assertEqual(k.content_type, phony_mimetype)
        k = bucket.lookup('notthere')
        self.assertIsNone(k)

    def test_metadata(self):
        """Test key metadata operations."""
        bucket = self._MakeBucket()
        k = self._MakeKey(bucket=bucket)
        key_name = k.name
        s1 = 'This is a test of file upload and download'
        mdkey1 = 'meta1'
        mdval1 = 'This is the first metadata value'
        k.set_metadata(mdkey1, mdval1)
        mdkey2 = 'meta2'
        mdval2 = 'This is the second metadata value'
        k.set_metadata(mdkey2, mdval2)
        mdval3 = u'föö'
        mdkey3 = 'meta3'
        k.set_metadata(mdkey3, mdval3)
        k.set_contents_from_string(s1)
        k = bucket.lookup(key_name)
        self.assertEqual(k.get_metadata(mdkey1), mdval1)
        self.assertEqual(k.get_metadata(mdkey2), mdval2)
        self.assertEqual(k.get_metadata(mdkey3), mdval3)
        k = bucket.new_key(key_name)
        k.get_contents_as_string()
        self.assertEqual(k.get_metadata(mdkey1), mdval1)
        self.assertEqual(k.get_metadata(mdkey2), mdval2)
        self.assertEqual(k.get_metadata(mdkey3), mdval3)

    def test_list_iterator(self):
        """Test list and iterator."""
        bucket = self._MakeBucket()
        num_iter = len([k for k in bucket.list()])
        rs = bucket.get_all_keys()
        num_keys = len(rs)
        self.assertEqual(num_iter, num_keys)

    def test_acl(self):
        """Test bucket and key ACLs."""
        bucket = self._MakeBucket()
        bucket.set_acl('public-read')
        acl = bucket.get_acl()
        self.assertEqual(len(acl.entries.entry_list), 2)
        bucket.set_acl('private')
        acl = bucket.get_acl()
        self.assertEqual(len(acl.entries.entry_list), 1)
        k = self._MakeKey(bucket=bucket)
        k.set_acl('public-read')
        acl = k.get_acl()
        self.assertEqual(len(acl.entries.entry_list), 2)
        k.set_acl('private')
        acl = k.get_acl()
        self.assertEqual(len(acl.entries.entry_list), 1)
        acl_xml = '<ACCESSControlList><EntrIes><Entry>' + '<Scope type="AllUsers"></Scope><Permission>READ</Permission>' + '</Entry></EntrIes></ACCESSControlList>'
        acl = ACL()
        h = handler.XmlHandler(acl, bucket)
        xml.sax.parseString(acl_xml, h)
        bucket.set_acl(acl)
        self.assertEqual(len(acl.entries.entry_list), 1)
        aclstr = k.get_xml_acl()
        self.assertGreater(aclstr.count('/Entry', 1), 0)

    def test_logging(self):
        """Test set/get raw logging subresource."""
        bucket = self._MakeBucket()
        empty_logging_str = "<?xml version='1.0' encoding='UTF-8'?><Logging/>"
        logging_str = "<?xml version='1.0' encoding='UTF-8'?><Logging><LogBucket>log-bucket</LogBucket>" + '<LogObjectPrefix>example</LogObjectPrefix>' + '</Logging>'
        bucket.set_subresource('logging', logging_str)
        self.assertEqual(bucket.get_subresource('logging'), logging_str)
        bucket.disable_logging()
        self.assertEqual(bucket.get_subresource('logging'), empty_logging_str)
        bucket.enable_logging('log-bucket', 'example')
        self.assertEqual(bucket.get_subresource('logging'), logging_str)

    def test_copy_key(self):
        """Test copying a key from one bucket to another."""
        bucket1 = self._MakeBucket()
        bucket2 = self._MakeBucket()
        bucket_name_1 = bucket1.name
        bucket_name_2 = bucket2.name
        bucket1 = self._GetConnection().get_bucket(bucket_name_1)
        bucket2 = self._GetConnection().get_bucket(bucket_name_2)
        key_name = 'foobar'
        k1 = bucket1.new_key(key_name)
        self.assertIsInstance(k1, bucket1.key_class)
        k1.name = key_name
        s = 'This is a test.'
        k1.set_contents_from_string(s)
        k1.copy(bucket_name_2, key_name)
        k2 = bucket2.lookup(key_name)
        self.assertIsInstance(k2, bucket2.key_class)
        tmpdir = self._MakeTempDir()
        fpath = os.path.join(tmpdir, 'foobar')
        fp = open(fpath, 'wb')
        k2.get_contents_to_file(fp)
        fp.close()
        fp = open(fpath)
        self.assertEqual(s, fp.read())
        fp.close()
        bucket1.delete_key(k1)
        bucket2.delete_key(k2)

    def test_default_object_acls(self):
        """Test default object acls."""
        bucket = self._MakeBucket()
        acl = bucket.get_def_acl()
        self.assertIsNotNone(re.search(PROJECT_PRIVATE_RE, acl.to_xml()))
        bucket.set_def_acl('public-read')
        acl = bucket.get_def_acl()
        public_read_acl = acl
        self.assertEqual(acl.to_xml(), '<AccessControlList><Entries><Entry><Scope type="AllUsers"></Scope><Permission>READ</Permission></Entry></Entries></AccessControlList>')
        bucket.set_def_acl('private')
        acl = bucket.get_def_acl()
        self.assertEqual(acl.to_xml(), '<AccessControlList></AccessControlList>')
        bucket.set_def_acl(public_read_acl)
        acl = bucket.get_def_acl()
        self.assertEqual(acl.to_xml(), '<AccessControlList><Entries><Entry><Scope type="AllUsers"></Scope><Permission>READ</Permission></Entry></Entries></AccessControlList>')
        bucket.set_def_acl('private')
        acl = bucket.get_def_acl()
        self.assertEqual(acl.to_xml(), '<AccessControlList></AccessControlList>')

    def test_default_object_acls_storage_uri(self):
        """Test default object acls using storage_uri."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        uri = storage_uri('gs://' + bucket_name)
        acl = uri.get_def_acl()
        self.assertIsNotNone(re.search(PROJECT_PRIVATE_RE, acl.to_xml()), 'PROJECT_PRIVATE_RE not found in ACL XML:\n' + acl.to_xml())
        uri.set_def_acl('public-read')
        acl = uri.get_def_acl()
        public_read_acl = acl
        self.assertEqual(acl.to_xml(), '<AccessControlList><Entries><Entry><Scope type="AllUsers"></Scope><Permission>READ</Permission></Entry></Entries></AccessControlList>')
        uri.set_def_acl('private')
        acl = uri.get_def_acl()
        self.assertEqual(acl.to_xml(), '<AccessControlList></AccessControlList>')
        uri.set_def_acl(public_read_acl)
        acl = uri.get_def_acl()
        self.assertEqual(acl.to_xml(), '<AccessControlList><Entries><Entry><Scope type="AllUsers"></Scope><Permission>READ</Permission></Entry></Entries></AccessControlList>')
        uri.set_def_acl('private')
        acl = uri.get_def_acl()
        self.assertEqual(acl.to_xml(), '<AccessControlList></AccessControlList>')

    def test_cors_xml_bucket(self):
        """Test setting and getting of CORS XML documents on Bucket."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        bucket = self._GetConnection().get_bucket(bucket_name)
        cors = re.sub('\\s', '', bucket.get_cors().to_xml())
        self.assertEqual(cors, CORS_EMPTY)
        bucket.set_cors(CORS_DOC)
        cors = re.sub('\\s', '', bucket.get_cors().to_xml())
        self.assertEqual(cors, CORS_DOC)

    def test_cors_xml_storage_uri(self):
        """Test setting and getting of CORS XML documents with storage_uri."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        uri = storage_uri('gs://' + bucket_name)
        cors = re.sub('\\s', '', uri.get_cors().to_xml())
        self.assertEqual(cors, CORS_EMPTY)
        cors_obj = Cors()
        h = handler.XmlHandler(cors_obj, None)
        xml.sax.parseString(CORS_DOC, h)
        uri.set_cors(cors_obj)
        cors = re.sub('\\s', '', uri.get_cors().to_xml())
        self.assertEqual(cors, CORS_DOC)

    def test_lifecycle_config_bucket(self):
        """Test setting and getting of lifecycle config on Bucket."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        bucket = self._GetConnection().get_bucket(bucket_name)
        xml = bucket.get_lifecycle_config().to_xml()
        self.assertEqual(xml, LIFECYCLE_EMPTY)
        lifecycle_config = LifecycleConfig()
        lifecycle_config.add_rule('Delete', None, LIFECYCLE_CONDITIONS_FOR_DELETE_RULE)
        lifecycle_config.add_rule('SetStorageClass', 'NEARLINE', LIFECYCLE_CONDITIONS_FOR_SET_STORAGE_CLASS_RULE)
        bucket.configure_lifecycle(lifecycle_config)
        xml = bucket.get_lifecycle_config().to_xml()
        self.assertEqual(xml, LIFECYCLE_DOC)

    def test_lifecycle_config_storage_uri(self):
        """Test setting and getting of lifecycle config with storage_uri."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        uri = storage_uri('gs://' + bucket_name)
        xml = uri.get_lifecycle_config().to_xml()
        self.assertEqual(xml, LIFECYCLE_EMPTY)
        lifecycle_config = LifecycleConfig()
        lifecycle_config.add_rule('Delete', None, LIFECYCLE_CONDITIONS_FOR_DELETE_RULE)
        lifecycle_config.add_rule('SetStorageClass', 'NEARLINE', LIFECYCLE_CONDITIONS_FOR_SET_STORAGE_CLASS_RULE)
        uri.configure_lifecycle(lifecycle_config)
        xml = uri.get_lifecycle_config().to_xml()
        self.assertEqual(xml, LIFECYCLE_DOC)

    def test_billing_config_bucket(self):
        """Test setting and getting of billing config on Bucket."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        billing = bucket.get_billing_config()
        self.assertEqual(billing, BILLING_EMPTY)
        bucket.configure_billing(requester_pays=True)
        billing = bucket.get_billing_config()
        self.assertEqual(billing, BILLING_ENABLED)
        bucket.configure_billing(requester_pays=False)
        billing = bucket.get_billing_config()
        self.assertEqual(billing, BILLING_DISABLED)

    def test_billing_config_storage_uri(self):
        """Test setting and getting of billing config with storage_uri."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        uri = storage_uri('gs://' + bucket_name)
        billing = uri.get_billing_config()
        self.assertEqual(billing, BILLING_EMPTY)
        uri.configure_billing(requester_pays=True)
        billing = uri.get_billing_config()
        self.assertEqual(billing, BILLING_ENABLED)
        uri.configure_billing(requester_pays=False)
        billing = uri.get_billing_config()
        self.assertEqual(billing, BILLING_DISABLED)

    def test_encryption_config_bucket(self):
        """Test setting and getting of EncryptionConfig on gs Bucket objects."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        encryption_config = bucket.get_encryption_config()
        self.assertIsNone(encryption_config.default_kms_key_name)
        xmldoc = bucket._construct_encryption_config_xml(default_kms_key_name='dummykey')
        self.assertEqual(xmldoc, ENCRYPTION_CONFIG_WITH_KEY % 'dummykey')
        bucket.set_encryption_config()

    def test_encryption_config_storage_uri(self):
        """Test setting and getting of EncryptionConfig with storage_uri."""
        bucket = self._MakeBucket()
        bucket_name = bucket.name
        uri = storage_uri('gs://' + bucket_name)
        encryption_config = uri.get_encryption_config()
        self.assertIsNone(encryption_config.default_kms_key_name)
        uri.set_encryption_config()