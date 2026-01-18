import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
class RecordsetCommands:

    def recordset_show(self, zone_id, id, *args, **kwargs):
        cmd = f'recordset show {zone_id} {id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def recordset_list(self, zone_id, *args, **kwargs):
        cmd = f'recordset list {zone_id}'
        return self.parsed_cmd(cmd, ListModel, *args, **kwargs)

    def recordset_create(self, zone_id, name, record=None, type=None, description=None, ttl=None, *args, **kwargs):
        options_str = build_option_string({'--record': record, '--type': type, '--description': description, '--ttl': ttl})
        cmd = f'recordset create {zone_id} {name} {options_str}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def recordset_set(self, zone_id, id, record=None, type=None, description=None, ttl=None, no_description=False, no_ttl=False, *args, **kwargs):
        options_str = build_option_string({'--record': record, '--type': type, '--description': description, '--ttl': ttl})
        flags_str = build_flags_string({'--no-description': no_description, '--no-ttl': no_ttl})
        cmd = f'recordset set {zone_id} {id} {flags_str} {options_str}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def recordset_delete(self, zone_id, id, *args, **kwargs):
        cmd = f'recordset delete {zone_id} {id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)