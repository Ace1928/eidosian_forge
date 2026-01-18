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
def DisableService(self, args):
    """Disable service of organization/folder/project."""
    if args.service == 'web-security-scanner':
        web_security_center_settings = self.message_module.WebSecurityScannerSettings(serviceEnablementState=self.message_module.WebSecurityScannerSettings.ServiceEnablementStateValueValuesEnum.DISABLED)
        return self._UpdateService(args, web_security_center_settings, SERVICE_STATUS_MASK)
    elif args.service == 'security-health-analytics':
        security_health_analytics_settings = self.message_module.SecurityHealthAnalyticsSettings(serviceEnablementState=self.message_module.SecurityHealthAnalyticsSettings.ServiceEnablementStateValueValuesEnum.DISABLED)
        return self._UpdateService(args, security_health_analytics_settings, SERVICE_STATUS_MASK)
    elif args.service == 'container-threat-detection':
        container_threat_detection_settings = self.message_module.ContainerThreatDetectionSettings(serviceEnablementState=self.message_module.ContainerThreatDetectionSettings.ServiceEnablementStateValueValuesEnum.DISABLED)
        return self._UpdateService(args, container_threat_detection_settings, SERVICE_STATUS_MASK)
    elif args.service == 'event-threat-detection':
        event_threat_detection_settings = self.message_module.EventThreatDetectionSettings(serviceEnablementState=self.message_module.EventThreatDetectionSettings.ServiceEnablementStateValueValuesEnum.DISABLED)
        return self._UpdateService(args, event_threat_detection_settings, SERVICE_STATUS_MASK)
    elif args.service == 'virtual-machine-threat-detection':
        virtual_machine_threat_detection_settings = self.message_module.VirtualMachineThreatDetectionSettings(serviceEnablementState=self.message_module.VirtualMachineThreatDetectionSettings.ServiceEnablementStateValueValuesEnum.DISABLED)
        return self._UpdateService(args, virtual_machine_threat_detection_settings, SERVICE_STATUS_MASK)
    elif args.service == 'rapid-vulnerability-detection':
        rapid_vulnerability_detection_settings = self.message_module.RapidVulnerabilityDetectionSettings(serviceEnablementState=self.message_module.RapidVulnerabilityDetectionSettings.ServiceEnablementStateValueValuesEnum.DISABLED)
        return self._UpdateService(args, rapid_vulnerability_detection_settings, SERVICE_STATUS_MASK)