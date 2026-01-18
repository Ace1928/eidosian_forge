from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import target_proxies_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.certificate_manager import resource_args
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute.backend_services import (
from googlecloudsdk.command_lib.compute.ssl_certificates import (
from googlecloudsdk.command_lib.compute.ssl_policies import (flags as
from googlecloudsdk.command_lib.compute.target_ssl_proxies import flags
from googlecloudsdk.command_lib.compute.target_ssl_proxies import target_ssl_proxies_utils
def _SendRequests(self, args, ssl_policy=None, clear_ssl_policy=False, certificate_map_ref=None):
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    requests = []
    target_ssl_proxy_ref = self.TARGET_SSL_PROXY_ARG.ResolveAsResource(args, holder.resources)
    client = holder.client.apitools_client
    messages = holder.client.messages
    clear_ssl_certificates = args.IsKnownAndSpecified('clear_ssl_certificates')
    if args.ssl_certificates or clear_ssl_certificates:
        ssl_certs = []
        if args.ssl_certificates:
            ssl_cert_refs = self.SSL_CERTIFICATES_ARG.ResolveAsResource(args, holder.resources)
            ssl_certs = [ref.SelfLink() for ref in ssl_cert_refs]
        requests.append((client.targetSslProxies, 'SetSslCertificates', messages.ComputeTargetSslProxiesSetSslCertificatesRequest(project=target_ssl_proxy_ref.project, targetSslProxy=target_ssl_proxy_ref.Name(), targetSslProxiesSetSslCertificatesRequest=messages.TargetSslProxiesSetSslCertificatesRequest(sslCertificates=ssl_certs))))
    if args.backend_service:
        backend_service_ref = self.BACKEND_SERVICE_ARG.ResolveAsResource(args, holder.resources)
        requests.append((client.targetSslProxies, 'SetBackendService', messages.ComputeTargetSslProxiesSetBackendServiceRequest(project=target_ssl_proxy_ref.project, targetSslProxy=target_ssl_proxy_ref.Name(), targetSslProxiesSetBackendServiceRequest=messages.TargetSslProxiesSetBackendServiceRequest(service=backend_service_ref.SelfLink()))))
    if args.proxy_header:
        proxy_header = messages.TargetSslProxiesSetProxyHeaderRequest.ProxyHeaderValueValuesEnum(args.proxy_header)
        requests.append((client.targetSslProxies, 'SetProxyHeader', messages.ComputeTargetSslProxiesSetProxyHeaderRequest(project=target_ssl_proxy_ref.project, targetSslProxy=target_ssl_proxy_ref.Name(), targetSslProxiesSetProxyHeaderRequest=messages.TargetSslProxiesSetProxyHeaderRequest(proxyHeader=proxy_header))))
    if args.ssl_policy:
        ssl_policy_ref = target_ssl_proxies_utils.ResolveSslPolicy(args, self.SSL_POLICY_ARG, target_ssl_proxy_ref, holder.resources)
        ssl_policy = messages.SslPolicyReference(sslPolicy=ssl_policy_ref.SelfLink())
    else:
        ssl_policy = None
    clear_ssl_policy = args.clear_ssl_policy
    if ssl_policy or clear_ssl_policy:
        requests.append((client.targetSslProxies, 'SetSslPolicy', messages.ComputeTargetSslProxiesSetSslPolicyRequest(project=target_ssl_proxy_ref.project, targetSslProxy=target_ssl_proxy_ref.Name(), sslPolicyReference=ssl_policy)))
    clear_certificate_map = args.IsKnownAndSpecified('clear_certificate_map')
    certificate_map_ref = args.CONCEPTS.certificate_map.Parse() if self._certificate_map else None
    if certificate_map_ref or clear_certificate_map:
        self_link = certificate_map_ref.SelfLink() if certificate_map_ref else None
        requests.append((client.targetSslProxies, 'SetCertificateMap', messages.ComputeTargetSslProxiesSetCertificateMapRequest(project=target_ssl_proxy_ref.project, targetSslProxy=target_ssl_proxy_ref.Name(), targetSslProxiesSetCertificateMapRequest=messages.TargetSslProxiesSetCertificateMapRequest(certificateMap=self_link))))
    errors = []
    resources = holder.client.MakeRequests(requests, errors)
    if errors:
        utils.RaiseToolException(errors)
    return resources