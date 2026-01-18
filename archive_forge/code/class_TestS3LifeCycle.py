from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
class TestS3LifeCycle(AWSMockServiceTestCase):
    connection_class = S3Connection

    def default_body(self):
        return '\n        <LifecycleConfiguration>\n          <Rule>\n            <ID>rule-1</ID>\n            <Prefix>prefix/foo</Prefix>\n            <Status>Enabled</Status>\n            <Transition>\n              <Days>30</Days>\n              <StorageClass>GLACIER</StorageClass>\n            </Transition>\n            <Expiration>\n              <Days>365</Days>\n            </Expiration>\n          </Rule>\n          <Rule>\n            <ID>rule-2</ID>\n            <Prefix>prefix/bar</Prefix>\n            <Status>Disabled</Status>\n            <Transition>\n              <Date>2012-12-31T00:00:000Z</Date>\n              <StorageClass>STANDARD_IA</StorageClass>\n            </Transition>\n            <Expiration>\n              <Date>2012-12-31T00:00:000Z</Date>\n            </Expiration>\n          </Rule>\n          <Rule>\n            <ID>multiple-transitions</ID>\n            <Prefix></Prefix>\n            <Status>Enabled</Status>\n            <Transition>\n                <Days>30</Days>\n                <StorageClass>STANDARD_IA</StorageClass>\n            </Transition>\n            <Transition>\n                <Days>90</Days>\n                <StorageClass>GLACIER</StorageClass>\n            </Transition>\n          </Rule>\n        </LifecycleConfiguration>\n        '

    def _get_bucket_lifecycle_config(self):
        self.set_http_response(status_code=200)
        bucket = Bucket(self.service_connection, 'mybucket')
        return bucket.get_lifecycle_config()

    def test_lifecycle_response_contains_all_rules(self):
        self.assertEqual(len(self._get_bucket_lifecycle_config()), 3)

    def test_parse_lifecycle_id(self):
        rule = self._get_bucket_lifecycle_config()[0]
        self.assertEqual(rule.id, 'rule-1')

    def test_parse_lifecycle_prefix(self):
        rule = self._get_bucket_lifecycle_config()[0]
        self.assertEqual(rule.prefix, 'prefix/foo')

    def test_parse_lifecycle_no_prefix(self):
        rule = self._get_bucket_lifecycle_config()[2]
        self.assertEquals(rule.prefix, '')

    def test_parse_lifecycle_enabled(self):
        rule = self._get_bucket_lifecycle_config()[0]
        self.assertEqual(rule.status, 'Enabled')

    def test_parse_lifecycle_disabled(self):
        rule = self._get_bucket_lifecycle_config()[1]
        self.assertEqual(rule.status, 'Disabled')

    def test_parse_expiration_days(self):
        rule = self._get_bucket_lifecycle_config()[0]
        self.assertEqual(rule.expiration.days, 365)

    def test_parse_expiration_date(self):
        rule = self._get_bucket_lifecycle_config()[1]
        self.assertEqual(rule.expiration.date, '2012-12-31T00:00:000Z')

    def test_parse_expiration_not_required(self):
        rule = self._get_bucket_lifecycle_config()[2]
        self.assertIsNone(rule.expiration)

    def test_parse_transition_days(self):
        transition = self._get_bucket_lifecycle_config()[0].transition[0]
        self.assertEquals(transition.days, 30)
        self.assertIsNone(transition.date)

    def test_parse_transition_days_deprecated(self):
        transition = self._get_bucket_lifecycle_config()[0].transition
        self.assertEquals(transition.days, 30)
        self.assertIsNone(transition.date)

    def test_parse_transition_date(self):
        transition = self._get_bucket_lifecycle_config()[1].transition[0]
        self.assertEquals(transition.date, '2012-12-31T00:00:000Z')
        self.assertIsNone(transition.days)

    def test_parse_transition_date_deprecated(self):
        transition = self._get_bucket_lifecycle_config()[1].transition
        self.assertEquals(transition.date, '2012-12-31T00:00:000Z')
        self.assertIsNone(transition.days)

    def test_parse_storage_class_standard_ia(self):
        transition = self._get_bucket_lifecycle_config()[1].transition[0]
        self.assertEqual(transition.storage_class, 'STANDARD_IA')

    def test_parse_storage_class_glacier(self):
        transition = self._get_bucket_lifecycle_config()[0].transition[0]
        self.assertEqual(transition.storage_class, 'GLACIER')

    def test_parse_storage_class_deprecated(self):
        transition = self._get_bucket_lifecycle_config()[1].transition
        self.assertEqual(transition.storage_class, 'STANDARD_IA')

    def test_parse_multiple_lifecycle_rules(self):
        transition = self._get_bucket_lifecycle_config()[2].transition
        self.assertEqual(len(transition), 2)

    def test_expiration_with_no_transition(self):
        lifecycle = Lifecycle()
        lifecycle.add_rule('myid', 'prefix', 'Enabled', 30)
        xml = lifecycle.to_xml()
        self.assertIn('<Expiration><Days>30</Days></Expiration>', xml)

    def test_expiration_is_optional(self):
        t = Transition(days=30, storage_class='GLACIER')
        r = Rule('myid', 'prefix', 'Enabled', expiration=None, transition=t)
        xml = r.to_xml()
        self.assertIn('<Transition><StorageClass>GLACIER</StorageClass><Days>30</Days>', xml)

    def test_transition_is_optional(self):
        r = Rule('myid', 'prefix', 'Enabled')
        xml = r.to_xml()
        self.assertEqual('<Rule><ID>myid</ID><Prefix>prefix</Prefix><Status>Enabled</Status></Rule>', xml)

    def test_expiration_and_transition(self):
        t = Transition(date='2012-11-30T00:00:000Z', storage_class='GLACIER')
        r = Rule('myid', 'prefix', 'Enabled', expiration=30, transition=t)
        xml = r.to_xml()
        self.assertIn('<Transition><StorageClass>GLACIER</StorageClass><Date>2012-11-30T00:00:000Z</Date>', xml)
        self.assertIn('<Expiration><Days>30</Days></Expiration>', xml)