from troveclient import base
from troveclient import common
class BackupStrategy(base.Resource):

    def __repr__(self):
        return '<BackupStrategy: %s[%s]>' % (self.project_id, self.instance_id)