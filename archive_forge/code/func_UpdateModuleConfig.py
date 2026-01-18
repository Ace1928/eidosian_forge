from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as gcloud_exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.scc.settings import exceptions as scc_exceptions
from googlecloudsdk.core import properties
def UpdateModuleConfig(self, args):
    """Update a config within a module."""
    if args.clear_config or args.config is None:
        config = None
    else:
        try:
            config = encoding.JsonToMessage(self.message_module.Config.ValueValue, args.config)
        except Exception:
            raise scc_exceptions.SecurityCenterSettingsException('Invalid argument {}. Check help text for an example json.'.format(args.config))
    enabled = args.enablement_state == 'enabled'
    return self._UpdateModules(args, enabled, args.clear_config, config)