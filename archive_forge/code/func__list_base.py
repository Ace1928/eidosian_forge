from novaclient import api_versions
from novaclient import base
def _list_base(self, host=None, status=None, instance_uuid=None, marker=None, limit=None, changes_since=None, changes_before=None, migration_type=None, source_compute=None, user_id=None, project_id=None):
    opts = {}
    if host:
        opts['host'] = host
    if status:
        opts['status'] = status
    if instance_uuid:
        opts['instance_uuid'] = instance_uuid
    if marker:
        opts['marker'] = marker
    if limit:
        opts['limit'] = limit
    if changes_since:
        opts['changes-since'] = changes_since
    if changes_before:
        opts['changes-before'] = changes_before
    if migration_type:
        opts['migration_type'] = migration_type
    if source_compute:
        opts['source_compute'] = source_compute
    if user_id:
        opts['user_id'] = user_id
    if project_id:
        opts['project_id'] = project_id
    return self._list('/os-migrations', 'migrations', filters=opts)