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
def _UpdateModules(self, args, enabled, clear_config=False, config=None):
    """Update modules within service settings."""
    StateEnum = self.message_module.Config.ModuleEnablementStateValueValuesEnum
    state = StateEnum.ENABLED if enabled else StateEnum.DISABLED
    curr_modules = None
    try:
        curr_modules = self.DescribeServiceExplicit(args).modules
    except gcloud_exceptions.HttpException as err:
        if err.payload.status_code == 404:
            curr_modules = None
            config = None
        else:
            raise err
    if not clear_config and config is None and (curr_modules is not None):
        module = [p for p in curr_modules.additionalProperties if p.key == args.module]
        if len(module) == 1:
            config = module[0].value.value
    if args.service == 'web-security-scanner':
        settings = self.message_module.WebSecurityScannerSettings(modules=self.message_module.WebSecurityScannerSettings.ModulesValue(additionalProperties=[self.message_module.WebSecurityScannerSettings.ModulesValue.AdditionalProperty(key=args.module, value=self.message_module.Config(moduleEnablementState=state, value=config))]))
    elif args.service == 'security-health-analytics':
        settings = self.message_module.SecurityHealthAnalyticsSettings(modules=self.message_module.SecurityHealthAnalyticsSettings.ModulesValue(additionalProperties=[self.message_module.SecurityHealthAnalyticsSettings.ModulesValue.AdditionalProperty(key=args.module, value=self.message_module.Config(moduleEnablementState=state, value=config))]))
    elif args.service == 'container-threat-detection':
        settings = self.message_module.ContainerThreatDetectionSettings(modules=self.message_module.ContainerThreatDetectionSettings.ModulesValue(additionalProperties=[self.message_module.ContainerThreatDetectionSettings.ModulesValue.AdditionalProperty(key=args.module, value=self.message_module.Config(moduleEnablementState=state, value=config))]))
    elif args.service == 'event-threat-detection':
        settings = self.message_module.EventThreatDetectionSettings(modules=self.message_module.EventThreatDetectionSettings.ModulesValue(additionalProperties=[self.message_module.EventThreatDetectionSettings.ModulesValue.AdditionalProperty(key=args.module, value=self.message_module.Config(moduleEnablementState=state, value=config))]))
    elif args.service == 'virtual-machine-threat-detection':
        settings = self.message_module.VirtualMachineThreatDetectionSettings(modules=self.message_module.VirtualMachineThreatDetectionSettings.ModulesValue(additionalProperties=[self.message_module.VirtualMachineThreatDetectionSettings.ModulesValue.AdditionalProperty(key=args.module, value=self.message_module.Config(moduleEnablementState=state, value=config))]))
    elif args.service == 'rapid-vulnerability-detection':
        settings = self.message_module.RapidVulnerabilityDetectionSettings(modules=self.message_module.RapidVulnerabilityDetectionSettings.ModulesValue(additionalProperties=[self.message_module.RapidVulnerabilityDetectionSettings.ModulesValue.AdditionalProperty(key=args.module, value=self.message_module.Config(moduleEnablementState=state, value=config))]))
    if curr_modules is not None:
        unmodified_additional_properties = [p for p in curr_modules.additionalProperties if p.key != args.module]
        settings.modules.additionalProperties = settings.modules.additionalProperties + unmodified_additional_properties
    return self._UpdateService(args, settings, MODULE_STATUS_MASK)