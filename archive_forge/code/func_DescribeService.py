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
def DescribeService(self, args):
    """Describe service settings of organization/folder/project."""
    FallBackFlags(args)
    path = GenerateParent(args) + SERVICES_ENDPOINTS[args.service]
    try:
        if args.organization:
            if args.service == 'web-security-scanner':
                request_message = self.message_module.SecuritycenterOrganizationsWebSecurityScannerSettingsCalculateRequest(name=path)
                return self.service_client.organizations_webSecurityScannerSettings.Calculate(request_message)
            elif args.service == 'security-health-analytics':
                request_message = self.message_module.SecuritycenterOrganizationsSecurityHealthAnalyticsSettingsCalculateRequest(name=path)
                return self.service_client.organizations_securityHealthAnalyticsSettings.Calculate(request_message)
            elif args.service == 'container-threat-detection':
                request_message = self.message_module.SecuritycenterOrganizationsContainerThreatDetectionSettingsCalculateRequest(name=path)
                return self.service_client.organizations_containerThreatDetectionSettings.Calculate(request_message)
            elif args.service == 'event-threat-detection':
                request_message = self.message_module.SecuritycenterOrganizationsEventThreatDetectionSettingsCalculateRequest(name=path)
                return self.service_client.organizations_eventThreatDetectionSettings.Calculate(request_message)
            elif args.service == 'virtual-machine-threat-detection':
                request_message = self.message_module.SecuritycenterOrganizationsVirtualMachineThreatDetectionSettingsCalculateRequest(name=path)
                return self.service_client.organizations_virtualMachineThreatDetectionSettings.Calculate(request_message)
            elif args.service == 'rapid-vulnerability-detection':
                request_message = self.message_module.SecuritycenterOrganizationsRapidVulnerabilityDetectionSettingsCalculateRequest(name=path)
                return self.service_client.organizations_rapidVulnerabilityDetectionSettings.Calculate(request_message)
        elif args.project:
            if args.service == 'web-security-scanner':
                request_message = self.message_module.SecuritycenterProjectsWebSecurityScannerSettingsCalculateRequest(name=path)
                return self.service_client.projects_webSecurityScannerSettings.Calculate(request_message)
            elif args.service == 'security-health-analytics':
                request_message = self.message_module.SecuritycenterProjectsSecurityHealthAnalyticsSettingsCalculateRequest(name=path)
                return self.service_client.projects_securityHealthAnalyticsSettings.Calculate(request_message)
            elif args.service == 'container-threat-detection':
                request_message = self.message_module.SecuritycenterProjectsContainerThreatDetectionSettingsCalculateRequest(name=path)
                return self.service_client.projects_containerThreatDetectionSettings.Calculate(request_message)
            elif args.service == 'event-threat-detection':
                request_message = self.message_module.SecuritycenterProjectsEventThreatDetectionSettingsCalculateRequest(name=path)
                return self.service_client.projects_eventThreatDetectionSettings.Calculate(request_message)
            elif args.service == 'virtual-machine-threat-detection':
                request_message = self.message_module.SecuritycenterProjectsVirtualMachineThreatDetectionSettingsCalculateRequest(name=path)
                return self.service_client.projects_virtualMachineThreatDetectionSettings.Calculate(request_message)
            elif args.service == 'rapid-vulnerability-detection':
                request_message = self.message_module.SecuritycenterProjectsRapidVulnerabilityDetectionSettingsCalculateRequest(name=path)
                return self.service_client.projects_rapidVulnerabilityDetectionSettings.Calculate(request_message)
        elif args.folder:
            if args.service == 'web-security-scanner':
                request_message = self.message_module.SecuritycenterFoldersWebSecurityScannerSettingsCalculateRequest(name=path)
                return self.service_client.folders_webSecurityScannerSettings.Calculate(request_message)
            elif args.service == 'security-health-analytics':
                request_message = self.message_module.SecuritycenterFoldersSecurityHealthAnalyticsSettingsCalculateRequest(name=path)
                return self.service_client.folders_securityHealthAnalyticsSettings.Calculate(request_message)
            elif args.service == 'container-threat-detection':
                request_message = self.message_module.SecuritycenterFoldersContainerThreatDetectionSettingsCalculateRequest(name=path)
                return self.service_client.folders_containerThreatDetectionSettings.Calculate(request_message)
            elif args.service == 'event-threat-detection':
                request_message = self.message_module.SecuritycenterFoldersEventThreatDetectionSettingsCalculateRequest(name=path)
                return self.service_client.folders_eventThreatDetectionSettings.Calculate(request_message)
            elif args.service == 'virtual-machine-threat-detection':
                request_message = self.message_module.SecuritycenterFoldersVirtualMachineThreatDetectionSettingsCalculateRequest(name=path)
                return self.service_client.folders_virtualMachineThreatDetectionSettings.Calculate(request_message)
            elif args.service == 'rapid-vulnerability-detection':
                request_message = self.message_module.SecuritycenterFoldersRapidVulnerabilityDetectionSettingsCalculateRequest(name=path)
                return self.service_client.folders_rapidVulnerabilityDetectionSettings.Calculate(request_message)
    except exceptions.HttpNotFoundError:
        raise scc_exceptions.SecurityCenterSettingsException('Invalid argument {}'.format(path))