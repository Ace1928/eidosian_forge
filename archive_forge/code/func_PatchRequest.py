from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter as container_api_adapter
from googlecloudsdk.api_lib.krmapihosting import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
def PatchRequest(args):
    """Construct a patch request based on the args."""
    instance = args.CONCEPTS.name.Parse()
    messages = apis.GetMessagesModule('krmapihosting', instance.GetCollectionInfo().api_version)
    current = util.GetKrmApiHost(instance.RelativeName())
    update_masks = []
    management_config = messages.ManagementConfig()
    bundles_config = messages.BundlesConfig(configControllerConfig=messages.ConfigControllerConfig())
    if args.experimental_features:
        update_masks.append('bundles_config.config_controller_config.experimental_features')
        bundles_config.configControllerConfig.experimentalFeatures = args.experimental_features
    if current.managementConfig.fullManagementConfig:
        full_management_config = messages.FullManagementConfig()
        if args.man_block:
            full_management_config.manBlock = args.man_block
            update_masks.append('management_config.full_management_config.man_block')
        management_config.fullManagementConfig = full_management_config
    else:
        standard_management_config = messages.StandardManagementConfig()
        if args.man_block:
            standard_management_config.manBlock = args.man_block
            update_masks.append('management_config.standard_management_config.man_block')
        management_config.standardManagementConfig = standard_management_config
    patch = messages.KrmApiHost(managementConfig=management_config, bundlesConfig=bundles_config)
    return messages.KrmapihostingProjectsLocationsKrmApiHostsPatchRequest(krmApiHost=patch, name=instance.RelativeName(), updateMask=','.join(update_masks))