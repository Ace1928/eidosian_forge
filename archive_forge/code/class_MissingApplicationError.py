from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class MissingApplicationError(exceptions.Error):
    """If an app does not exist within the current project."""

    def __init__(self, project):
        self.project = project

    def __str__(self):
        return 'The current Google Cloud project [{0}] does not contain an App Engine application. Use `gcloud app create` to initialize an App Engine application within the project.'.format(self.project)