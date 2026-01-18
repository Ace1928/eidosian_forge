from magnumclient.common import base
class StatsManager(base.Manager):
    resource_class = Stats

    @staticmethod
    def _path(id=None):
        return '/v1/stats?project_id=%s' % id if id else '/v1/stats'

    def list(self, project_id=None):
        try:
            return self._list(self._path(project_id))[0]
        except IndexError:
            return None