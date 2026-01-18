import unittest
from boto.s3.cors import CORSConfiguration
class TestCORSConfiguration(unittest.TestCase):

    def test_one_rule_with_id(self):
        cfg = CORSConfiguration()
        cfg.add_rule(['PUT', 'POST', 'DELETE'], 'http://www.example.com', allowed_header='*', max_age_seconds=3000, expose_header='x-amz-server-side-encryption', id='foobar_rule')
        self.assertEqual(cfg.to_xml(), CORS_BODY_1)

    def test_two_rules(self):
        cfg = CORSConfiguration()
        cfg.add_rule(['PUT', 'POST', 'DELETE'], 'http://www.example.com', allowed_header='*', max_age_seconds=3000, expose_header='x-amz-server-side-encryption')
        cfg.add_rule('GET', '*', allowed_header='*', max_age_seconds=3000)
        self.assertEqual(cfg.to_xml(), CORS_BODY_2)

    def test_minimal(self):
        cfg = CORSConfiguration()
        cfg.add_rule('GET', '*')
        self.assertEqual(cfg.to_xml(), CORS_BODY_3)