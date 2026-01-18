import collections
import configparser
import re
from oslo_log import log as logging
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import eventfactory as factory
from pycadf import host
from pycadf import identifier
from pycadf import resource
from pycadf import tag
from urllib import parse as urlparse
class ClientResource(resource.Resource):

    def __init__(self, project_id=None, request_id=None, global_request_id=None, **kwargs):
        super(ClientResource, self).__init__(**kwargs)
        if project_id is not None:
            self.project_id = project_id
        if request_id is not None:
            self.request_id = request_id
        if global_request_id is not None:
            self.global_request_id = global_request_id