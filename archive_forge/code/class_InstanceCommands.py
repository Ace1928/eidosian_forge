import os
import sys
from troveclient.compat import common
class InstanceCommands(common.AuthedCommandsBase):
    """Commands to perform various instance operations and actions."""
    params = ['flavor', 'id', 'limit', 'marker', 'name', 'size', 'backup', 'availability_zone', 'configuration_id']

    def _get_configuration_ref(self):
        configuration_ref = None
        if self.configuration_id is not None:
            if self.configuration_id == '':
                configuration_ref = self.configuration_id
            else:
                configuration_ref = '/'.join([self.dbaas.client.service_url, self.configuration_id])
        return configuration_ref

    def create(self):
        """Create a new instance."""
        self._require('name', 'flavor')
        volume = None
        if self.size:
            volume = {'size': self.size}
        restorePoint = None
        if self.backup:
            restorePoint = {'backupRef': self.backup}
        self._pretty_print(self.dbaas.instances.create, self.name, self.flavor, volume, restorePoint=restorePoint, availability_zone=self.availability_zone, configuration=self._get_configuration_ref())

    def modify(self):
        """Modify an instance."""
        self._require('id')
        self._pretty_print(self.dbaas.instances.modify, self.id, configuration=self._get_configuration_ref())

    def delete(self):
        """Delete the specified instance."""
        self._require('id')
        print(self.dbaas.instances.delete(self.id))

    def get(self):
        """Get details for the specified instance."""
        self._require('id')
        self._pretty_print(self.dbaas.instances.get, self.id)

    def backups(self):
        """Get a list of backups for the specified instance."""
        self._require('id')
        self._pretty_list(self.dbaas.instances.backups, self.id)

    def list(self):
        """List all instances for account."""
        limit = self.limit or None
        if limit:
            limit = int(limit, 10)
        self._pretty_paged(self.dbaas.instances.list)

    def resize_volume(self):
        """Resize an instance volume."""
        self._require('id', 'size')
        self._pretty_print(self.dbaas.instances.resize_volume, self.id, self.size)

    def resize_instance(self):
        """Resize an instance flavor"""
        self._require('id', 'flavor')
        self._pretty_print(self.dbaas.instances.resize_instance, self.id, self.flavor)

    def restart(self):
        """Restart the database."""
        self._require('id')
        self._pretty_print(self.dbaas.instances.restart, self.id)

    def configuration(self):
        """Get configuration for the specified instance."""
        self._require('id')
        self._pretty_print(self.dbaas.instances.configuration, self.id)