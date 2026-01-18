from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import target_proxies_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.certificate_manager import resource_args
from googlecloudsdk.command_lib.compute.backend_services import (
from googlecloudsdk.command_lib.compute.ssl_certificates import (
from googlecloudsdk.command_lib.compute.ssl_policies import (flags as
from googlecloudsdk.command_lib.compute.target_ssl_proxies import flags
from googlecloudsdk.command_lib.compute.target_ssl_proxies import target_ssl_proxies_utils
def _CreateResource(self, args):
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    backend_service_ref = self.BACKEND_SERVICE_ARG.ResolveAsResource(args, holder.resources)
    target_ssl_proxy_ref = self.TARGET_SSL_PROXY_ARG.ResolveAsResource(args, holder.resources)
    ssl_cert_refs = None
    if args.ssl_certificates:
        ssl_cert_refs = self.SSL_CERTIFICATES_ARG.ResolveAsResource(args, holder.resources)
    client = holder.client.apitools_client
    messages = holder.client.messages
    if args.proxy_header:
        proxy_header = messages.TargetSslProxy.ProxyHeaderValueValuesEnum(args.proxy_header)
    else:
        proxy_header = messages.TargetSslProxy.ProxyHeaderValueValuesEnum.NONE
    target_ssl_proxy = messages.TargetSslProxy(description=args.description, name=target_ssl_proxy_ref.Name(), proxyHeader=proxy_header, service=backend_service_ref.SelfLink())
    if ssl_cert_refs:
        target_ssl_proxy.sslCertificates = [ref.SelfLink() for ref in ssl_cert_refs]
    if args.ssl_policy:
        target_ssl_proxy.sslPolicy = target_ssl_proxies_utils.ResolveSslPolicy(args, self.SSL_POLICY_ARG, target_ssl_proxy_ref, holder.resources).SelfLink()
    if self._certificate_map:
        certificate_map_ref = args.CONCEPTS.certificate_map.Parse()
        if certificate_map_ref:
            target_ssl_proxy.certificateMap = certificate_map_ref.SelfLink()
    request = messages.ComputeTargetSslProxiesInsertRequest(project=target_ssl_proxy_ref.project, targetSslProxy=target_ssl_proxy)
    errors = []
    resources = holder.client.MakeRequests([(client.targetSslProxies, 'Insert', request)], errors)
    if errors:
        utils.RaiseToolException(errors)
    return resources