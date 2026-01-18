from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
def _get_resources(self, parsed_args):
    mistral_client = self.app.client_manager.workflow_engine
    return mistral_client.event_triggers.list(marker=parsed_args.marker, limit=parsed_args.limit, sort_keys=parsed_args.sort_keys, sort_dirs=parsed_args.sort_dirs, fields=EventTriggerFormatter.fields(), **base.get_filters(parsed_args))