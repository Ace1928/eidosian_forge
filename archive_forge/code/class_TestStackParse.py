import datetime
import xml.sax
import unittest
import boto.handler
import boto.resultset
import boto.cloudformation
class TestStackParse(unittest.TestCase):

    def test_parse_tags(self):
        rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
        h = boto.handler.XmlHandler(rs, None)
        xml.sax.parseString(SAMPLE_XML, h)
        tags = rs[0].tags
        self.assertEqual(tags, {u'key0': u'value0', u'key1': u'value1'})

    def test_event_creation_time_with_millis(self):
        millis_xml = SAMPLE_XML.replace(b'<CreationTime>2013-01-10T05:04:56Z</CreationTime>', b'<CreationTime>2013-01-10T05:04:56.102342Z</CreationTime>')
        rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
        h = boto.handler.XmlHandler(rs, None)
        xml.sax.parseString(millis_xml, h)
        creation_time = rs[0].creation_time
        self.assertEqual(creation_time, datetime.datetime(2013, 1, 10, 5, 4, 56, 102342))

    def test_resource_time_with_millis(self):
        rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.StackResource)])
        h = boto.handler.XmlHandler(rs, None)
        xml.sax.parseString(DESCRIBE_STACK_RESOURCE_XML, h)
        timestamp_1 = rs[0].timestamp
        self.assertEqual(timestamp_1, datetime.datetime(2010, 7, 27, 22, 27, 28))
        timestamp_2 = rs[1].timestamp
        self.assertEqual(timestamp_2, datetime.datetime(2010, 7, 27, 22, 28, 28, 123456))

    def test_list_stacks_time_with_millis(self):
        rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.StackSummary)])
        h = boto.handler.XmlHandler(rs, None)
        xml.sax.parseString(LIST_STACKS_XML, h)
        timestamp_1 = rs[0].creation_time
        self.assertEqual(timestamp_1, datetime.datetime(2011, 5, 23, 15, 47, 44))
        timestamp_2 = rs[1].creation_time
        self.assertEqual(timestamp_2, datetime.datetime(2011, 3, 5, 19, 57, 58, 161616))
        timestamp_3 = rs[1].deletion_time
        self.assertEqual(timestamp_3, datetime.datetime(2011, 3, 10, 16, 20, 51, 575757))

    def test_list_stacks_time_with_millis_again(self):
        rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.StackResourceSummary)])
        h = boto.handler.XmlHandler(rs, None)
        xml.sax.parseString(LIST_STACK_RESOURCES_XML, h)
        timestamp_1 = rs[0].last_updated_time
        self.assertEqual(timestamp_1, datetime.datetime(2011, 6, 21, 20, 15, 58))
        timestamp_2 = rs[1].last_updated_time
        self.assertEqual(timestamp_2, datetime.datetime(2011, 6, 21, 20, 25, 57, 875643))

    def test_disable_rollback_false(self):
        rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
        h = boto.handler.XmlHandler(rs, None)
        xml.sax.parseString(SAMPLE_XML, h)
        disable_rollback = rs[0].disable_rollback
        self.assertFalse(disable_rollback)

    def test_disable_rollback_false_upper(self):
        rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
        h = boto.handler.XmlHandler(rs, None)
        sample_xml_upper = SAMPLE_XML.replace(b'false', b'False')
        xml.sax.parseString(sample_xml_upper, h)
        disable_rollback = rs[0].disable_rollback
        self.assertFalse(disable_rollback)

    def test_disable_rollback_true(self):
        rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
        h = boto.handler.XmlHandler(rs, None)
        sample_xml_upper = SAMPLE_XML.replace(b'false', b'true')
        xml.sax.parseString(sample_xml_upper, h)
        disable_rollback = rs[0].disable_rollback
        self.assertTrue(disable_rollback)

    def test_disable_rollback_true_upper(self):
        rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
        h = boto.handler.XmlHandler(rs, None)
        sample_xml_upper = SAMPLE_XML.replace(b'false', b'True')
        xml.sax.parseString(sample_xml_upper, h)
        disable_rollback = rs[0].disable_rollback
        self.assertTrue(disable_rollback)