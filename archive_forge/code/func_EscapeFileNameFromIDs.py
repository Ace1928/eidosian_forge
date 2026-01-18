from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.core import resources
def EscapeFileNameFromIDs(project_id, location_id, repo_id, file_id):
    """Escapes slashes and pluses from request names."""
    return resources.REGISTRY.Create('artifactregistry.projects.locations.repositories.files', projectsId=project_id, locationsId=location_id, repositoriesId=repo_id, filesId=file_id.replace('/', '%2F').replace('+', '%2B').replace('^', '%5E'))