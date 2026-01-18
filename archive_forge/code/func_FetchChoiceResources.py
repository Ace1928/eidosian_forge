from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import gce as c_gce
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def FetchChoiceResources(self, attribute, service, flag_names, prefix_filter=None):
    """Returns a list of choices used to prompt with."""
    if prefix_filter:
        filter_expr = 'name eq {0}.*'.format(prefix_filter)
    else:
        filter_expr = None
    errors = []
    global_resources = lister.GetGlobalResources(service=service, project=self.project, filter_expr=filter_expr, http=self.http, batch_url=self.batch_url, errors=errors)
    choices = [resource for resource in global_resources]
    if errors or not choices:
        punctuation = ':' if errors else '.'
        utils.RaiseToolException(errors, 'Unable to fetch a list of {0}s. Specifying [{1}] may fix this issue{2}'.format(attribute, ', or '.join(flag_names), punctuation))
    return choices