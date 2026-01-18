import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
class CommandsBaseTest(testtools.TestCase):

    def setUp(self):
        super(CommandsBaseTest, self).setUp()
        self.orig_sys_exit = sys.exit
        sys.exit = mock.Mock(return_value=None)
        self.orig_sys_argv = sys.argv
        sys.argv = ['fakecmd']
        parser = common.CliOptions().create_optparser(False)
        self.cmd_base = common.CommandsBase(parser)

    def tearDown(self):
        super(CommandsBaseTest, self).tearDown()
        sys.exit = self.orig_sys_exit
        sys.argv = self.orig_sys_argv

    def test___init__(self):
        self.assertIsNotNone(self.cmd_base)

    def test__safe_exec(self):
        func = mock.Mock(return_value='test')
        self.cmd_base.debug = True
        r = self.cmd_base._safe_exec(func)
        self.assertEqual('test', r)
        self.cmd_base.debug = False
        r = self.cmd_base._safe_exec(func)
        self.assertEqual('test', r)
        func = mock.Mock(side_effect=ValueError)
        r = self.cmd_base._safe_exec(func)
        self.assertIsNone(r)

    def test__prepare_parser(self):
        parser = optparse.OptionParser()
        common.CommandsBase.params = ['test_1', 'test_2']
        self.cmd_base._prepare_parser(parser)
        option = parser.get_option('--%s' % 'test_1')
        self.assertIsNotNone(option)
        option = parser.get_option('--%s' % 'test_2')
        self.assertIsNotNone(option)

    def test__parse_options(self):
        parser = optparse.OptionParser()
        parser.add_option('--%s' % 'test_1', default='test_1v')
        parser.add_option('--%s' % 'test_2', default='test_2v')
        self.cmd_base._parse_options(parser)
        self.assertEqual('test_1v', self.cmd_base.test_1)
        self.assertEqual('test_2v', self.cmd_base.test_2)

    def test__require(self):
        self.assertRaises(common.ArgumentRequired, self.cmd_base._require, 'attr_1')
        self.cmd_base.attr_1 = None
        self.assertRaises(common.ArgumentRequired, self.cmd_base._require, 'attr_1')
        self.cmd_base.attr_1 = 'attr_v1'
        self.cmd_base._require('attr_1')

    def test__make_list(self):
        self.assertRaises(AttributeError, self.cmd_base._make_list, 'attr1')
        self.cmd_base.attr1 = 'v1,v2'
        self.cmd_base._make_list('attr1')
        self.assertEqual(['v1', 'v2'], self.cmd_base.attr1)
        self.cmd_base.attr1 = ['v3']
        self.cmd_base._make_list('attr1')
        self.assertEqual(['v3'], self.cmd_base.attr1)

    def test__pretty_print(self):
        func = mock.Mock(return_value=None)
        self.cmd_base.verbose = True
        self.assertIsNone(self.cmd_base._pretty_print(func))
        self.cmd_base.verbose = False
        self.assertIsNone(self.cmd_base._pretty_print(func))

    def test__dumps(self):
        orig_dumps = json.dumps
        json.dumps = mock.Mock(return_value='test-dump')
        self.assertEqual('test-dump', self.cmd_base._dumps('item'))
        json.dumps = orig_dumps

    def test__pretty_list(self):
        func = mock.Mock(return_value=None)
        self.cmd_base.verbose = True
        self.assertIsNone(self.cmd_base._pretty_list(func))
        self.cmd_base.verbose = False
        self.assertIsNone(self.cmd_base._pretty_list(func))
        item = mock.Mock(return_value='test')
        item._info = 'info'
        func = mock.Mock(return_value=[item])
        self.assertIsNone(self.cmd_base._pretty_list(func))

    def test__pretty_paged(self):
        self.cmd_base.limit = '5'
        func = mock.Mock(return_value=None)
        self.cmd_base.verbose = True
        self.assertIsNone(self.cmd_base._pretty_paged(func))
        self.cmd_base.verbose = False

        class MockIterable(collections.abc.Iterable):
            links = ['item']
            count = 1

            def __iter__(self):
                return ['item1']

            def __len__(self):
                return self.count
        ret = MockIterable()
        func = mock.Mock(return_value=ret)
        self.assertIsNone(self.cmd_base._pretty_paged(func))
        ret.count = 0
        self.assertIsNone(self.cmd_base._pretty_paged(func))
        func = mock.Mock(side_effect=ValueError)
        self.assertIsNone(self.cmd_base._pretty_paged(func))
        self.cmd_base.debug = True
        self.cmd_base.marker = mock.Mock()
        self.assertRaises(ValueError, self.cmd_base._pretty_paged, func)