from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import utils
def GetHostProject(self, project):
    """Get the XPN host for the given project."""
    request_tuple = (self.client.projects, 'GetXpnHost', self.messages.ComputeProjectsGetXpnHostRequest(project=project))
    msg = 'get XPN host for project [{project}]'.format(project=project)
    return self._MakeRequestSync(request_tuple, msg)