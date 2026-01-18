import textwrap
from unittest import mock
from oslotest import base
from oslo_config import cfg
from oslo_config import sphinxext
class FormatOptionHelpTest(base.BaseTestCase):

    @mock.patch('oslo_config.generator._list_opts')
    @mock.patch('oslo_config.sphinxext._format_group_opts')
    def test_split_namespaces(self, _format_group_opts, _list_opts):
        _list_opts.return_value = [('namespace1', [(None, ['opt1'])]), ('namespace2', [(None, ['opt2'])])]
        list(sphinxext._format_option_help(namespaces=['namespace1', 'namespace2'], split_namespaces=True))
        _format_group_opts.assert_any_call(namespace='namespace1', group_name='DEFAULT', group_obj=None, opt_list=['opt1'])
        _format_group_opts.assert_any_call(namespace='namespace2', group_name='DEFAULT', group_obj=None, opt_list=['opt2'])

    @mock.patch('oslo_config.generator._list_opts')
    @mock.patch('oslo_config.sphinxext._format_group_opts')
    def test_dont_split_namespaces(self, _format_group_opts, _list_opts):
        _list_opts.return_value = [('namespace1', [(None, ['opt1'])]), ('namespace2', [(None, ['opt2'])])]
        list(sphinxext._format_option_help(namespaces=['namespace1', 'namespace2'], split_namespaces=False))
        _format_group_opts.assert_called_once_with(namespace=None, group_name='DEFAULT', group_obj=None, opt_list=['opt1', 'opt2'])

    @mock.patch('oslo_config.generator._list_opts')
    @mock.patch('oslo_config.sphinxext._format_group_opts')
    def test_dont_split_namespaces_with_group(self, _format_group_opts, _list_opts):
        grp_obj = cfg.OptGroup('grp1')
        _list_opts.return_value = [('namespace1', [(grp_obj, ['opt1'])]), ('namespace2', [('grp1', ['opt2'])])]
        list(sphinxext._format_option_help(namespaces=['namespace1', 'namespace2'], split_namespaces=False))
        _format_group_opts.assert_any_call(namespace=None, group_name='grp1', group_obj=grp_obj, opt_list=['opt1', 'opt2'])

    @mock.patch('oslo_config.generator._list_opts')
    @mock.patch('oslo_config.sphinxext._format_group_opts')
    def test_split_namespaces_with_group(self, _format_group_opts, _list_opts):
        grp_obj = cfg.OptGroup('grp1')
        _list_opts.return_value = [('namespace1', [(grp_obj, ['opt1'])]), ('namespace2', [('grp1', ['opt2'])])]
        list(sphinxext._format_option_help(namespaces=['namespace1', 'namespace2'], split_namespaces=True))
        print(_format_group_opts.call_args_list)
        _format_group_opts.assert_any_call(namespace='namespace1', group_name='grp1', group_obj=grp_obj, opt_list=['opt1'])
        _format_group_opts.assert_any_call(namespace='namespace2', group_name='grp1', group_obj=None, opt_list=['opt2'])