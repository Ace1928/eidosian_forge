from troveclient import base
from troveclient import common
class BackupStrategiesManager(base.ManagerWithFind):
    resource_class = BackupStrategy

    def list(self, instance_id=None, project_id=None):
        query_strings = {}
        if instance_id:
            query_strings['instance_id'] = instance_id
        if project_id:
            query_strings['project_id'] = project_id
        url = common.append_query_strings('/backup_strategies', **query_strings)
        return self._list(url, 'backup_strategies')

    def create(self, instance_id=None, swift_container=None):
        backup_strategy = {}
        if instance_id:
            backup_strategy['instance_id'] = instance_id
        if swift_container:
            backup_strategy['swift_container'] = swift_container
        body = {'backup_strategy': backup_strategy}
        return self._create('/backup_strategies', body, 'backup_strategy')

    def delete(self, instance_id=None, project_id=None):
        url = '/backup_strategies'
        query_strings = {}
        if instance_id:
            query_strings['instance_id'] = instance_id
        if project_id:
            query_strings['project_id'] = project_id
        url = common.append_query_strings('/backup_strategies', **query_strings)
        resp, body = self._delete(url)
        common.check_for_exceptions(resp, body, url)