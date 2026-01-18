import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
class ZoneCommands:
    """This is a mixin that provides zone commands to DesignateCLI"""

    def zone_list(self, *args, **kwargs):
        return self.parsed_cmd('zone list', ListModel, *args, **kwargs)

    def zone_show(self, id, *args, **kwargs):
        return self.parsed_cmd(f'zone show {id}', FieldValueModel, *args, **kwargs)

    def zone_delete(self, id, *args, **kwargs):
        return self.parsed_cmd(f'zone delete {id}', FieldValueModel, *args, **kwargs)

    def zone_create(self, name, email=None, ttl=None, description=None, type=None, masters=None, *args, **kwargs):
        options_str = build_option_string({'--email': email, '--ttl': ttl, '--description': description, '--masters': masters, '--type': type})
        cmd = f'zone create {name} {options_str}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_set(self, id, email=None, ttl=None, description=None, type=None, masters=None, *args, **kwargs):
        options_str = build_option_string({'--email': email, '--ttl': ttl, '--description': description, '--masters': masters, '--type': type})
        cmd = f'zone set {id} {options_str}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)