from oslo_serialization import jsonutils
from mistralclient.api import base
class ActionExecution(base.Resource):
    resource_name = 'ActionExecution'