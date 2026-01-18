from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ProtoToManifest(resource, resource_ref, kind):
    """Converts a resource message to a cloud deploy resource manifest.

  The manifest can be applied by 'deploy apply' command.

  Args:
    resource: message in googlecloudsdk.generated_clients.apis.clouddeploy.
    resource_ref: cloud deploy resource object.
    kind: kind of the cloud deploy resource

  Returns:
    A dictionary that represents the cloud deploy resource.
  """
    manifest = collections.OrderedDict(apiVersion=API_VERSION_V1, kind=kind, metadata={})
    for k in METADATA_FIELDS:
        v = getattr(resource, k)
        if v:
            manifest['metadata'][k] = v
    if kind == AUTOMATION_KIND:
        manifest['metadata'][NAME_FIELD] = resource_ref.AsDict()['deliveryPipelinesId'] + '/' + resource_ref.Name()
    else:
        manifest['metadata'][NAME_FIELD] = resource_ref.Name()
    for f in resource.all_fields():
        if f.name in EXCLUDE_FIELDS:
            continue
        v = getattr(resource, f.name)
        if v:
            if f.name == SELECTOR_FIELD and kind == AUTOMATION_KIND:
                ExportAutomationSelector(manifest, v)
                continue
            if f.name == RULES_FIELD and kind == AUTOMATION_KIND:
                ExportAutomationRules(manifest, v)
                continue
            manifest[f.name] = v
    return manifest