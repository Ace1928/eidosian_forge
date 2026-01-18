import copy
import logging
import socket
import botocore.exceptions
import botocore.parsers
import botocore.serialize
from botocore.config import Config
from botocore.endpoint import EndpointCreator
from botocore.regions import EndpointResolverBuiltins as EPRBuiltins
from botocore.regions import EndpointRulesetResolver
from botocore.signers import RequestSigner
from botocore.useragent import UserAgentString
from botocore.utils import ensure_boolean, is_s3_accelerate_url
def _build_endpoint_resolver(self, endpoints_ruleset_data, partition_data, client_config, service_model, endpoint_region_name, region_name, endpoint_url, endpoint, is_secure, endpoint_bridge, event_emitter):
    if endpoints_ruleset_data is None:
        return None
    s3_config_raw = self.compute_s3_config(client_config) or {}
    service_name_raw = service_model.endpoint_prefix
    if service_name_raw in ['s3', 'sts'] or region_name is None:
        eprv2_region_name = endpoint_region_name
    else:
        eprv2_region_name = region_name
    resolver_builtins = self.compute_endpoint_resolver_builtin_defaults(region_name=eprv2_region_name, service_name=service_name_raw, s3_config=s3_config_raw, endpoint_bridge=endpoint_bridge, client_endpoint_url=endpoint_url, legacy_endpoint_url=endpoint.host)
    if client_config is not None:
        client_context = client_config.client_context_params or {}
    else:
        client_context = {}
    if self._is_s3_service(service_name_raw):
        client_context.update(s3_config_raw)
    sig_version = client_config.signature_version if client_config is not None else None
    return EndpointRulesetResolver(endpoint_ruleset_data=endpoints_ruleset_data, partition_data=partition_data, service_model=service_model, builtins=resolver_builtins, client_context=client_context, event_emitter=event_emitter, use_ssl=is_secure, requested_auth_scheme=sig_version)