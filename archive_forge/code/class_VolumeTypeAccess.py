from cinderclient.apiclient import base as common_base
from cinderclient import base
class VolumeTypeAccess(base.Resource):

    def __repr__(self):
        return '<VolumeTypeAccess: %s>' % self.project_id