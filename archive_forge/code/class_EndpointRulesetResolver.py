import copy
import logging
import re
from enum import Enum
from botocore import UNSIGNED, xform_name
from botocore.auth import AUTH_TYPE_MAPS, HAS_CRT
from botocore.crt import CRT_SUPPORTED_AUTH_TYPES
from botocore.endpoint_provider import EndpointProvider
from botocore.exceptions import (
from botocore.utils import ensure_boolean, instance_cache
class EndpointRulesetResolver:
    """Resolves endpoints using a service's endpoint ruleset"""

    def __init__(self, endpoint_ruleset_data, partition_data, service_model, builtins, client_context, event_emitter, use_ssl=True, requested_auth_scheme=None):
        self._provider = EndpointProvider(ruleset_data=endpoint_ruleset_data, partition_data=partition_data)
        self._param_definitions = self._provider.ruleset.parameters
        self._service_model = service_model
        self._builtins = builtins
        self._client_context = client_context
        self._event_emitter = event_emitter
        self._use_ssl = use_ssl
        self._requested_auth_scheme = requested_auth_scheme
        self._instance_cache = {}

    def construct_endpoint(self, operation_model, call_args, request_context):
        """Invokes the provider with params defined in the service's ruleset"""
        if call_args is None:
            call_args = {}
        if request_context is None:
            request_context = {}
        provider_params = self._get_provider_params(operation_model, call_args, request_context)
        LOG.debug('Calling endpoint provider with parameters: %s' % provider_params)
        try:
            provider_result = self._provider.resolve_endpoint(**provider_params)
        except EndpointProviderError as ex:
            botocore_exception = self.ruleset_error_to_botocore_exception(ex, provider_params)
            if botocore_exception is None:
                raise
            else:
                raise botocore_exception from ex
        LOG.debug('Endpoint provider result: %s' % provider_result.url)
        if not self._use_ssl and provider_result.url.startswith('https://'):
            provider_result = provider_result._replace(url=f'http://{provider_result.url[8:]}')
        provider_result = provider_result._replace(headers={key: val[0] for key, val in provider_result.headers.items()})
        return provider_result

    def _get_provider_params(self, operation_model, call_args, request_context):
        """Resolve a value for each parameter defined in the service's ruleset

        The resolution order for parameter values is:
        1. Operation-specific static context values from the service definition
        2. Operation-specific dynamic context values from API parameters
        3. Client-specific context parameters
        4. Built-in values such as region, FIPS usage, ...
        """
        provider_params = {}
        customized_builtins = self._get_customized_builtins(operation_model, call_args, request_context)
        for param_name, param_def in self._param_definitions.items():
            param_val = self._resolve_param_from_context(param_name=param_name, operation_model=operation_model, call_args=call_args)
            if param_val is None and param_def.builtin is not None:
                param_val = self._resolve_param_as_builtin(builtin_name=param_def.builtin, builtins=customized_builtins)
            if param_val is not None:
                provider_params[param_name] = param_val
        return provider_params

    def _resolve_param_from_context(self, param_name, operation_model, call_args):
        static = self._resolve_param_as_static_context_param(param_name, operation_model)
        if static is not None:
            return static
        dynamic = self._resolve_param_as_dynamic_context_param(param_name, operation_model, call_args)
        if dynamic is not None:
            return dynamic
        return self._resolve_param_as_client_context_param(param_name)

    def _resolve_param_as_static_context_param(self, param_name, operation_model):
        static_ctx_params = self._get_static_context_params(operation_model)
        return static_ctx_params.get(param_name)

    def _resolve_param_as_dynamic_context_param(self, param_name, operation_model, call_args):
        dynamic_ctx_params = self._get_dynamic_context_params(operation_model)
        if param_name in dynamic_ctx_params:
            member_name = dynamic_ctx_params[param_name]
            return call_args.get(member_name)

    def _resolve_param_as_client_context_param(self, param_name):
        client_ctx_params = self._get_client_context_params()
        if param_name in client_ctx_params:
            client_ctx_varname = client_ctx_params[param_name]
            return self._client_context.get(client_ctx_varname)

    def _resolve_param_as_builtin(self, builtin_name, builtins):
        if builtin_name not in EndpointResolverBuiltins.__members__.values():
            raise UnknownEndpointResolutionBuiltInName(name=builtin_name)
        return builtins.get(builtin_name)

    @instance_cache
    def _get_static_context_params(self, operation_model):
        """Mapping of param names to static param value for an operation"""
        return {param.name: param.value for param in operation_model.static_context_parameters}

    @instance_cache
    def _get_dynamic_context_params(self, operation_model):
        """Mapping of param names to member names for an operation"""
        return {param.name: param.member_name for param in operation_model.context_parameters}

    @instance_cache
    def _get_client_context_params(self):
        """Mapping of param names to client configuration variable"""
        return {param.name: xform_name(param.name) for param in self._service_model.client_context_parameters}

    def _get_customized_builtins(self, operation_model, call_args, request_context):
        service_id = self._service_model.service_id.hyphenize()
        customized_builtins = copy.copy(self._builtins)
        self._event_emitter.emit('before-endpoint-resolution.%s' % service_id, builtins=customized_builtins, model=operation_model, params=call_args, context=request_context)
        return customized_builtins

    def auth_schemes_to_signing_ctx(self, auth_schemes):
        """Convert an Endpoint's authSchemes property to a signing_context dict

        :type auth_schemes: list
        :param auth_schemes: A list of dictionaries taken from the
            ``authSchemes`` property of an Endpoint object returned by
            ``EndpointProvider``.

        :rtype: str, dict
        :return: Tuple of auth type string (to be used in
            ``request_context['auth_type']``) and signing context dict (for use
            in ``request_context['signing']``).
        """
        if not isinstance(auth_schemes, list) or len(auth_schemes) == 0:
            raise TypeError('auth_schemes must be a non-empty list.')
        LOG.debug('Selecting from endpoint provider\'s list of auth schemes: %s. User selected auth scheme is: "%s"', ', '.join([f'"{s.get('name')}"' for s in auth_schemes]), self._requested_auth_scheme)
        if self._requested_auth_scheme == UNSIGNED:
            return ('none', {})
        auth_schemes = [{**scheme, 'name': self._strip_sig_prefix(scheme['name'])} for scheme in auth_schemes]
        if self._requested_auth_scheme is not None:
            try:
                name, scheme = next(((self._requested_auth_scheme, s) for s in auth_schemes if self._does_botocore_authname_match_ruleset_authname(self._requested_auth_scheme, s['name'])))
            except StopIteration:
                return (None, {})
        else:
            try:
                name, scheme = next(((s['name'], s) for s in auth_schemes if s['name'] in AUTH_TYPE_MAPS))
            except StopIteration:
                fixable_with_crt = False
                auth_type_options = [s['name'] for s in auth_schemes]
                if not HAS_CRT:
                    fixable_with_crt = any((scheme in CRT_SUPPORTED_AUTH_TYPES for scheme in auth_type_options))
                if fixable_with_crt:
                    raise MissingDependencyException(msg='This operation requires an additional dependency. Use pip install botocore[crt] before proceeding.')
                else:
                    raise UnknownSignatureVersionError(signature_version=', '.join(auth_type_options))
        signing_context = {}
        if 'signingRegion' in scheme:
            signing_context['region'] = scheme['signingRegion']
        elif 'signingRegionSet' in scheme:
            if len(scheme['signingRegionSet']) > 0:
                signing_context['region'] = scheme['signingRegionSet'][0]
        if 'signingName' in scheme:
            signing_context.update(signing_name=scheme['signingName'])
        if 'disableDoubleEncoding' in scheme:
            signing_context['disableDoubleEncoding'] = ensure_boolean(scheme['disableDoubleEncoding'])
        LOG.debug('Selected auth type "%s" as "%s" with signing context params: %s', scheme['name'], name, signing_context)
        return (name, signing_context)

    def _strip_sig_prefix(self, auth_name):
        """Normalize auth type names by removing any "sig" prefix"""
        return auth_name[3:] if auth_name.startswith('sig') else auth_name

    def _does_botocore_authname_match_ruleset_authname(self, botoname, rsname):
        """
        Whether a valid string provided as signature_version parameter for
        client construction refers to the same auth methods as a string
        returned by the endpoint ruleset provider. This accounts for:

        * The ruleset prefixes auth names with "sig"
        * The s3 and s3control rulesets don't distinguish between v4[a] and
          s3v4[a] signers
        * The v2, v3, and HMAC v1 based signers (s3, s3-*) are botocore legacy
          features and do not exist in the rulesets
        * Only characters up to the first dash are considered

        Example matches:
        * v4, sigv4
        * v4, v4
        * s3v4, sigv4
        * s3v7, sigv7 (hypothetical example)
        * s3v4a, sigv4a
        * s3v4-query, sigv4

        Example mismatches:
        * v4a, sigv4
        * s3, sigv4
        * s3-presign-post, sigv4
        """
        rsname = self._strip_sig_prefix(rsname)
        botoname = botoname.split('-')[0]
        if botoname != 's3' and botoname.startswith('s3'):
            botoname = botoname[2:]
        return rsname == botoname

    def ruleset_error_to_botocore_exception(self, ruleset_exception, params):
        """Attempts to translate ruleset errors to pre-existing botocore
        exception types by string matching exception strings.
        """
        msg = ruleset_exception.kwargs.get('msg')
        if msg is None:
            return
        if msg.startswith('Invalid region in ARN: '):
            try:
                label = msg.split('`')[1]
            except IndexError:
                label = msg
            return InvalidHostLabelError(label=label)
        service_name = self._service_model.service_name
        if service_name == 's3':
            if msg == 'S3 Object Lambda does not support S3 Accelerate' or msg == 'Accelerate cannot be used with FIPS':
                return UnsupportedS3ConfigurationError(msg=msg)
            if msg.startswith('S3 Outposts does not support') or msg.startswith('S3 MRAP does not support') or msg.startswith('S3 Object Lambda does not support') or msg.startswith('Access Points do not support') or msg.startswith('Invalid configuration:') or msg.startswith('Client was configured for partition'):
                return UnsupportedS3AccesspointConfigurationError(msg=msg)
            if msg.lower().startswith('invalid arn:'):
                return ParamValidationError(report=msg)
        if service_name == 's3control':
            if msg.startswith('Invalid ARN:'):
                arn = params.get('Bucket')
                return UnsupportedS3ControlArnError(arn=arn, msg=msg)
            if msg.startswith('Invalid configuration:') or msg.startswith('Client was configured for partition'):
                return UnsupportedS3ControlConfigurationError(msg=msg)
            if msg == 'AccountId is required but not set':
                return ParamValidationError(report=msg)
        if service_name == 'events':
            if msg.startswith('Invalid Configuration: FIPS is not supported with EventBridge multi-region endpoints.'):
                return InvalidEndpointConfigurationError(msg=msg)
            if msg == 'EndpointId must be a valid host label.':
                return InvalidEndpointConfigurationError(msg=msg)
        return None