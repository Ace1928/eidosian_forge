from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import Union
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1 import gkehub_v1_client as ga_client
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_client as alpha_client
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as alpha_messages
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_client as beta_client
def RBACRoleBindingParentName(project, namespace, release_track=base.ReleaseTrack.ALPHA):
    return resources.REGISTRY.Parse(line=None, params={'projectsId': project, 'locationsId': 'global', 'namespacesId': namespace}, collection='gkehub.projects.locations.namespaces', api_version=VERSION_MAP[release_track]).RelativeName()