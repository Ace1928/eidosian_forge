from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def _FindKubernetesApiName(domain):
    """Find the name of the kubernetes api.

  Determines the kubernetes api name from the domain of the resource uri.
  The domain may from a global resource or a regionalized resource.

  Args:
    domain: Domain from the resource uri.

  Returns:
    Api name. Returns None if the domain is not a kubernetes api domain.
  """
    k8s_api_names = ('run',)
    domain_first_part = domain.split('.')[0]
    for api_name in k8s_api_names:
        if api_name == domain_first_part or domain_first_part.endswith('-' + api_name):
            return api_name
    return None