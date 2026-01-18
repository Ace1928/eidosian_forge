import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
class ZoneTransferCommands:
    """A mixin for DesignateCLI to add zone transfer commands"""

    def zone_transfer_request_list(self, *args, **kwargs):
        cmd = 'zone transfer request list'
        return self.parsed_cmd(cmd, ListModel, *args, **kwargs)

    def zone_transfer_request_create(self, zone_id, target_project_id=None, description=None, *args, **kwargs):
        options_str = build_option_string({'--target-project-id': target_project_id, '--description': description})
        cmd = f'zone transfer request create {zone_id} {options_str}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_transfer_request_show(self, id, *args, **kwargs):
        cmd = f'zone transfer request show {id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_transfer_request_set(self, id, description=None, *args, **kwargs):
        options_str = build_option_string({'--description': description})
        cmd = f'zone transfer request set {options_str} {id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_transfer_request_delete(self, id, *args, **kwargs):
        cmd = f'zone transfer request delete {id}'
        return self.parsed_cmd(cmd, *args, **kwargs)

    def zone_transfer_accept_request(self, id, key, *args, **kwargs):
        options_str = build_option_string({'--transfer-id': id, '--key': key})
        cmd = f'zone transfer accept request {options_str}'
        return self.parsed_cmd(cmd, *args, **kwargs)

    def zone_transfer_accept_show(self, id, *args, **kwargs):
        cmd = f'zone transfer accept show {id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_transfer_accept_list(self, *args, **kwargs):
        cmd = 'zone transfer accept list'
        return self.parsed_cmd(cmd, ListModel, *args, **kwargs)