from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import acm_printer
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def ExpandLevelNamesIfNecessary(level_ids, policy_id):
    """Returns the FULL Access Level names, prepending Policy ID if necessary."""
    if level_ids is None:
        return None
    final_level_ids = []
    for l in level_ids:
        if l.startswith('accessPolicies/'):
            final_level_ids.append(l)
        else:
            final_level_ids.append(REGISTRY.Create(levels.COLLECTION, accessPoliciesId=policy_id, accessLevelsId=l))
    return final_level_ids