from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def GetMatchingImages(self, user_project, image, alias, errors):
    """Yields images from a public image project and the user's project."""
    service = self._compute.images
    requests = [(service, 'List', self._messages.ComputeImagesListRequest(filter='name eq ^{0}(-.+)*-v.+'.format(alias.name_prefix), maxResults=constants.MAX_RESULTS_PER_PAGE, project=alias.project)), (service, 'List', self._messages.ComputeImagesListRequest(filter='name eq ^{0}$'.format(image), maxResults=constants.MAX_RESULTS_PER_PAGE, project=user_project))]
    return request_helper.MakeRequests(requests=requests, http=self._http, batch_url=self._batch_url, errors=errors)