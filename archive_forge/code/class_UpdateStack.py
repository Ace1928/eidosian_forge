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
class UpdateStack(command.ShowOne):
    """Update a stack."""
    log = logging.getLogger(__name__ + '.UpdateStack')

    def get_parser(self, prog_name):
        parser = super(UpdateStack, self).get_parser(prog_name)
        parser.add_argument('-t', '--template', metavar='<template>', help=_('Path to the template'))
        parser.add_argument('-s', '--files-container', metavar='<files-container>', help=_('Swift files container name. Local files other than root template would be ignored. If other files are not found in swift, heat engine would raise an error.'))
        parser.add_argument('-e', '--environment', metavar='<environment>', action='append', help=_('Path to the environment. Can be specified multiple times'))
        parser.add_argument('--pre-update', metavar='<resource>', action='append', help=_('Name of a resource to set a pre-update hook to. Resources in nested stacks can be set using slash as a separator: ``nested_stack/another/my_resource``. You can use wildcards to match multiple stacks or resources: ``nested_stack/an*/*_resource``. This can be specified multiple times'))
        parser.add_argument('--timeout', metavar='<timeout>', type=int, help=_('Stack update timeout in minutes'))
        parser.add_argument('--rollback', metavar='<value>', help=_('Set rollback on update failure. Value "enabled" sets rollback to enabled. Value "disabled" sets rollback to disabled. Value "keep" uses the value of existing stack to be updated (default)'))
        parser.add_argument('--dry-run', action='store_true', help=_('Do not actually perform the stack update, but show what would be changed'))
        parser.add_argument('--show-nested', default=False, action='store_true', help=_('Show nested stacks when performing --dry-run'))
        parser.add_argument('--parameter', metavar='<key=value>', help=_('Parameter values used to create the stack. This can be specified multiple times'), action='append')
        parser.add_argument('--parameter-file', metavar='<key=file>', help=_('Parameter values from file used to create the stack. This can be specified multiple times. Parameter value would be the content of the file'), action='append')
        parser.add_argument('--existing', action='store_true', help=_('Re-use the template, parameters and environment of the current stack. If the template argument is omitted then the existing template is used. If no %(env_arg)s is specified then the existing environment is used. Parameters specified in %(arg)s will patch over the existing values in the current stack. Parameters omitted will keep the existing values') % {'arg': '--parameter', 'env_arg': '--environment'})
        parser.add_argument('--clear-parameter', metavar='<parameter>', help=_('Remove the parameters from the set of parameters of current stack for the %(cmd)s. The default value in the template will be used. This can be specified multiple times') % {'cmd': 'stack-update'}, action='append')
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack to update'))
        parser.add_argument('--tags', metavar='<tag1,tag2...>', help=_('An updated list of tags to associate with the stack'))
        parser.add_argument('--wait', action='store_true', help=_('Wait until stack goes to UPDATE_COMPLETE or UPDATE_FAILED'))
        parser.add_argument('--converge', action='store_true', help=_('Stack update with observe on reality.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        tpl_files, template = template_utils.process_template_path(parsed_args.template, object_request=http.authenticated_fetcher(client), existing=parsed_args.existing, fetch_child=parsed_args.files_container is None)
        env_files_list = []
        env_files, env = template_utils.process_multiple_environments_and_files(env_paths=parsed_args.environment, env_list_tracker=env_files_list, fetch_env_files=parsed_args.files_container is None)
        parameters = heat_utils.format_all_parameters(parsed_args.parameter, parsed_args.parameter_file, parsed_args.template)
        if parsed_args.pre_update:
            template_utils.hooks_to_env(env, parsed_args.pre_update, 'pre-update')
        fields = {'stack_id': parsed_args.stack, 'parameters': parameters, 'existing': parsed_args.existing, 'template': template, 'files': dict(list(tpl_files.items()) + list(env_files.items())), 'environment': env}
        if env_files_list:
            fields['environment_files'] = env_files_list
        if parsed_args.files_container:
            fields['files_container'] = parsed_args.files_container
        if parsed_args.tags:
            fields['tags'] = parsed_args.tags
        if parsed_args.timeout:
            fields['timeout_mins'] = parsed_args.timeout
        if parsed_args.clear_parameter:
            fields['clear_parameters'] = list(parsed_args.clear_parameter)
        if parsed_args.rollback:
            rollback = parsed_args.rollback.strip().lower()
            if rollback not in ('enabled', 'disabled', 'keep'):
                msg = _('--rollback invalid value: %s') % parsed_args.rollback
                raise exc.CommandError(msg)
            if rollback != 'keep':
                fields['disable_rollback'] = rollback == 'disabled'
        if parsed_args.dry_run:
            if parsed_args.show_nested:
                fields['show_nested'] = parsed_args.show_nested
            changes = client.stacks.preview_update(**fields)
            fields = ['state', 'resource_name', 'resource_type', 'resource_identity']
            columns = sorted(changes.get('resource_changes', {}).keys())
            data = [heat_utils.json_formatter(changes['resource_changes'][key]) for key in columns]
            return (columns, data)
        if parsed_args.wait:
            events = event_utils.get_events(client, stack_id=parsed_args.stack, event_args={'sort_dir': 'desc'}, limit=1)
            marker = events[0].id if events else None
        if parsed_args.converge:
            fields['converge'] = True
        client.stacks.update(**fields)
        if parsed_args.wait:
            stack = client.stacks.get(parsed_args.stack)
            stack_status, msg = event_utils.poll_for_events(client, stack.stack_name, action='UPDATE', marker=marker)
            if stack_status == 'UPDATE_FAILED':
                raise exc.CommandError(msg)
        return _show_stack(client, parsed_args.stack, format='table', short=True)