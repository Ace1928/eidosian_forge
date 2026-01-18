from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
class DeletedResource(object):
    """A deleted/undeleted resource returned by Run()."""

    def __init__(self, project_id):
        self.projectId = project_id