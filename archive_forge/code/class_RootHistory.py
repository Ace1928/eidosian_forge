import json
from troveclient import base
from troveclient import common
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
class RootHistory(base.Resource):

    def __repr__(self):
        return '<Root History: Instance %s enabled at %s by %s>' % (self.id, self.created, self.user)