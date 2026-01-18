import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
class ZoneExportCommands:
    """A mixin for DesignateCLI to add zone export commands"""

    def zone_export_list(self, *args, **kwargs):
        cmd = 'zone export list'
        return self.parsed_cmd(cmd, ListModel, *args, **kwargs)

    def zone_export_create(self, zone_id, *args, **kwargs):
        cmd = f'zone export create {zone_id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_export_show(self, zone_export_id, *args, **kwargs):
        cmd = f'zone export show {zone_export_id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)

    def zone_export_delete(self, zone_export_id, *args, **kwargs):
        cmd = f'zone export delete {zone_export_id}'
        return self.parsed_cmd(cmd, *args, **kwargs)

    def zone_export_showfile(self, zone_export_id, *args, **kwargs):
        cmd = f'zone export showfile {zone_export_id}'
        return self.parsed_cmd(cmd, FieldValueModel, *args, **kwargs)