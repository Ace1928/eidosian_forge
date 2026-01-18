from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
class CreateCapsule(command.ShowOne):
    """Create a capsule"""
    log = logging.getLogger(__name__ + '.CreateCapsule')

    def get_parser(self, prog_name):
        parser = super(CreateCapsule, self).get_parser(prog_name)
        parser.add_argument('--file', metavar='<template_file>', required=True, help='Path to the capsule template file.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['template'] = template_utils.get_template_contents(parsed_args.file)
        capsule = client.capsules.create(**opts)
        columns = _capsule_columns(capsule)
        return (columns, utils.get_item_properties(capsule, columns))