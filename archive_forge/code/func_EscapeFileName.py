from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.core import resources
def EscapeFileName(ref):
    """Escapes slashes and pluses from request names."""
    return resources.REGISTRY.Create('artifactregistry.projects.locations.repositories.files', projectsId=ref.projectsId, locationsId=ref.locationsId, repositoriesId=ref.repositoriesId, filesId=ref.filesId.replace('/', '%2F').replace('+', '%2B').replace('^', '%5E'))