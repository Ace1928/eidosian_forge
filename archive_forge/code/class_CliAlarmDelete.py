import argparse
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from aodhclient import exceptions
from aodhclient.i18n import _
from aodhclient import utils
class CliAlarmDelete(command.Command):
    """Delete an alarm"""

    def get_parser(self, prog_name):
        return _add_name_to_parser(_add_id_to_parser(super(CliAlarmDelete, self).get_parser(prog_name)))

    def take_action(self, parsed_args):
        _check_name_and_id(parsed_args, 'delete')
        c = utils.get_client(self)
        if parsed_args.name:
            _id = _find_alarm_id_by_name(c, parsed_args.name)
        elif uuidutils.is_uuid_like(parsed_args.id):
            try:
                return c.alarm.delete(parsed_args.id)
            except exceptions.NotFound:
                _id = _find_alarm_id_by_name(c, parsed_args.id)
        else:
            _id = _find_alarm_id_by_name(c, parsed_args.id)
        c.alarm.delete(_id)