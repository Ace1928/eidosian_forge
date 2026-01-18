import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import event_utils
from heatclient.common import format_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class CreateStack(command.ShowOne):
    """Create a stack."""
    log = logging.getLogger(__name__ + '.CreateStack')

    def get_parser(self, prog_name):
        parser = super(CreateStack, self).get_parser(prog_name)
        parser.add_argument('-e', '--environment', metavar='<environment>', action='append', help=_('Path to the environment. Can be specified multiple times'))
        parser.add_argument('-s', '--files-container', metavar='<files-container>', help=_('Swift files container name. Local files other than root template would be ignored. If other files are not found in swift, heat engine would raise an error.'))
        parser.add_argument('--timeout', metavar='<timeout>', type=int, help=_('Stack creating timeout in minutes'))
        parser.add_argument('--pre-create', metavar='<resource>', default=None, action='append', help=_('Name of a resource to set a pre-create hook to. Resources in nested stacks can be set using slash as a separator: ``nested_stack/another/my_resource``. You can use wildcards to match multiple stacks or resources: ``nested_stack/an*/*_resource``. This can be specified multiple times'))
        parser.add_argument('--enable-rollback', action='store_true', help=_('Enable rollback on create/update failure'))
        parser.add_argument('--parameter', metavar='<key=value>', action='append', help=_('Parameter values used to create the stack. This can be specified multiple times'))
        parser.add_argument('--parameter-file', metavar='<key=file>', action='append', help=_('Parameter values from file used to create the stack. This can be specified multiple times. Parameter values would be the content of the file'))
        parser.add_argument('--wait', action='store_true', help=_('Wait until stack goes to CREATE_COMPLETE or CREATE_FAILED'))
        parser.add_argument('--poll', metavar='SECONDS', type=int, default=5, help=_('Poll interval in seconds for use with --wait, defaults to 5.'))
        parser.add_argument('--tags', metavar='<tag1,tag2...>', help=_('A list of tags to associate with the stack'))
        parser.add_argument('--dry-run', action='store_true', help=_('Do not actually perform the stack create, but show what would be created'))
        parser.add_argument('name', metavar='<stack-name>', help=_('Name of the stack to create'))
        parser.add_argument('-t', '--template', metavar='<template>', required=True, help=_('Path to the template'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        tpl_files, template = template_utils.process_template_path(parsed_args.template, object_request=http.authenticated_fetcher(client), fetch_child=parsed_args.files_container is None)
        env_files_list = []
        env_files, env = template_utils.process_multiple_environments_and_files(env_paths=parsed_args.environment, env_list_tracker=env_files_list, fetch_env_files=parsed_args.files_container is None)
        parameters = heat_utils.format_all_parameters(parsed_args.parameter, parsed_args.parameter_file, parsed_args.template)
        if parsed_args.pre_create:
            template_utils.hooks_to_env(env, parsed_args.pre_create, 'pre-create')
        fields = {'stack_name': parsed_args.name, 'disable_rollback': not parsed_args.enable_rollback, 'parameters': parameters, 'template': template, 'files': dict(list(tpl_files.items()) + list(env_files.items())), 'environment': env}
        if env_files_list:
            fields['environment_files'] = env_files_list
        if parsed_args.files_container:
            fields['files_container'] = parsed_args.files_container
        if parsed_args.tags:
            fields['tags'] = parsed_args.tags
        if parsed_args.timeout:
            fields['timeout_mins'] = parsed_args.timeout
        if parsed_args.dry_run:
            stack = client.stacks.preview(**fields)
            formatters = {'description': heat_utils.text_wrap_formatter, 'template_description': heat_utils.text_wrap_formatter, 'stack_status_reason': heat_utils.text_wrap_formatter, 'parameters': heat_utils.json_formatter, 'outputs': heat_utils.json_formatter, 'resources': heat_utils.json_formatter, 'links': heat_utils.link_formatter}
            columns = []
            for key in stack.to_dict():
                columns.append(key)
            columns.sort()
            return (columns, utils.get_item_properties(stack, columns, formatters=formatters))
        stack = client.stacks.create(**fields)['stack']
        if parsed_args.wait:
            stack_status, msg = event_utils.poll_for_events(client, parsed_args.name, action='CREATE', poll_period=parsed_args.poll)
            if stack_status == 'CREATE_FAILED':
                raise exc.CommandError(msg)
        return _show_stack(client, stack['id'], format='table', short=True)