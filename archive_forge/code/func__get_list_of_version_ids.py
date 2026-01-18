import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
def _get_list_of_version_ids(self, status):
    request = webob.Request.blank('/')
    request.accept = 'application/json'
    response = versions.Controller().index(request)
    v_list = jsonutils.loads(response.body)['versions']
    return [v['id'] for v in v_list if v['status'] == status]