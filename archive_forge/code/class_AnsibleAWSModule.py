import logging
import os
import re
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import boto3_at_least
from .botocore import boto3_conn
from .botocore import botocore_at_least
from .botocore import check_sdk_version_supported
from .botocore import gather_sdk_versions
from .botocore import get_aws_connection_info
from .botocore import get_aws_region
from .exceptions import AnsibleBotocoreError
from .retries import RetryingBotoClientWrapper
class AnsibleAWSModule:
    """An ansible module class for AWS modules

    AnsibleAWSModule provides an a class for building modules which
    connect to Amazon Web Services.  The interface is currently more
    restricted than the basic module class with the aim that later the
    basic module class can be reduced.  If you find that any key
    feature is missing please contact the author/Ansible AWS team
    (available on #ansible-aws on IRC) to request the additional
    features needed.
    """
    default_settings = {'default_args': True, 'check_boto3': True, 'auto_retry': True, 'module_class': AnsibleModule}

    def __init__(self, **kwargs):
        local_settings = {}
        for key in AnsibleAWSModule.default_settings:
            try:
                local_settings[key] = kwargs.pop(key)
            except KeyError:
                local_settings[key] = AnsibleAWSModule.default_settings[key]
        self.settings = local_settings
        if local_settings['default_args']:
            argument_spec_full = aws_argument_spec()
            try:
                argument_spec_full.update(kwargs['argument_spec'])
            except (TypeError, NameError):
                pass
            kwargs['argument_spec'] = argument_spec_full
        self._module = AnsibleAWSModule.default_settings['module_class'](**kwargs)
        if local_settings['check_boto3']:
            try:
                check_sdk_version_supported(warn=self.warn)
            except AnsibleBotocoreError as e:
                self._module.fail_json(to_native(e))
        deprecated_vars = {'EC2_REGION', 'EC2_SECURITY_TOKEN', 'EC2_SECRET_KEY', 'EC2_ACCESS_KEY', 'EC2_URL', 'S3_URL'}
        if deprecated_vars.intersection(set(os.environ.keys())):
            self._module.deprecate("Support for the 'EC2_REGION', 'EC2_ACCESS_KEY', 'EC2_SECRET_KEY', 'EC2_SECURITY_TOKEN', 'EC2_URL', and 'S3_URL' environment variables has been deprecated.  These variables are currently used for all AWS services which can cause confusion.  We recomend using the relevant module parameters or alternatively the 'AWS_REGION', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', and 'AWS_URL' environment variables can be used instead.", date='2024-12-01', collection_name='amazon.aws')
        if 'AWS_SECURITY_TOKEN' in os.environ.keys():
            self._module.deprecate("Support for the 'AWS_SECURITY_TOKEN' environment variable has been deprecated.  This variable was based on the original boto SDK, support for which has now been dropped.  We recommend using the 'session_token' module parameter or alternatively the 'AWS_SESSION_TOKEN' environment variable can be used instead.", date='2024-12-01', collection_name='amazon.aws')
        self.check_mode = self._module.check_mode
        self._diff = self._module._diff
        self._name = self._module._name
        self._botocore_endpoint_log_stream = StringIO()
        self.logger = None
        if self.params.get('debug_botocore_endpoint_logs'):
            self.logger = logging.getLogger('botocore.endpoint')
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(logging.StreamHandler(self._botocore_endpoint_log_stream))

    @property
    def params(self):
        return self._module.params

    def _get_resource_action_list(self):
        actions = []
        for ln in self._botocore_endpoint_log_stream.getvalue().split('\n'):
            ln = ln.strip()
            if not ln:
                continue
            found_operational_request = re.search('OperationModel\\(name=.*?\\)', ln)
            if found_operational_request:
                operation_request = found_operational_request.group(0)[20:-1]
                resource = re.search('https://.*?\\.', ln).group(0)[8:-1]
                actions.append(f'{resource}:{operation_request}')
        return list(set(actions))

    def exit_json(self, *args, **kwargs):
        if self.params.get('debug_botocore_endpoint_logs'):
            kwargs['resource_actions'] = self._get_resource_action_list()
        return self._module.exit_json(*args, **kwargs)

    def fail_json(self, *args, **kwargs):
        if self.params.get('debug_botocore_endpoint_logs'):
            kwargs['resource_actions'] = self._get_resource_action_list()
        return self._module.fail_json(*args, **kwargs)

    def debug(self, *args, **kwargs):
        return self._module.debug(*args, **kwargs)

    def warn(self, *args, **kwargs):
        return self._module.warn(*args, **kwargs)

    def deprecate(self, *args, **kwargs):
        return self._module.deprecate(*args, **kwargs)

    def boolean(self, *args, **kwargs):
        return self._module.boolean(*args, **kwargs)

    def md5(self, *args, **kwargs):
        return self._module.md5(*args, **kwargs)

    def client(self, service, retry_decorator=None, **extra_params):
        region, endpoint_url, aws_connect_kwargs = get_aws_connection_info(self, boto3=True)
        kw_args = dict(region=region, endpoint=endpoint_url, **aws_connect_kwargs)
        kw_args.update(extra_params)
        conn = boto3_conn(self, conn_type='client', resource=service, **kw_args)
        return conn if retry_decorator is None else RetryingBotoClientWrapper(conn, retry_decorator)

    def resource(self, service, **extra_params):
        region, endpoint_url, aws_connect_kwargs = get_aws_connection_info(self, boto3=True)
        kw_args = dict(region=region, endpoint=endpoint_url, **aws_connect_kwargs)
        kw_args.update(extra_params)
        return boto3_conn(self, conn_type='resource', resource=service, **kw_args)

    @property
    def region(self):
        return get_aws_region(self, True)

    def fail_json_aws(self, exception, msg=None, **kwargs):
        """call fail_json with processed exception

        function for converting exceptions thrown by AWS SDK modules,
        botocore, boto3 and boto, into nice error messages.
        """
        last_traceback = traceback.format_exc()
        try:
            except_msg = to_native(exception.message)
        except AttributeError:
            except_msg = to_native(exception)
        if msg is not None:
            message = f'{msg}: {except_msg}'
        else:
            message = except_msg
        try:
            response = exception.response
        except AttributeError:
            response = None
        failure = dict(msg=message, exception=last_traceback, **self._gather_versions())
        failure.update(kwargs)
        if response is not None:
            failure.update(**camel_dict_to_snake_dict(response))
        self.fail_json(**failure)

    def fail_json_aws_error(self, exception):
        """A helper to call the right failure mode after catching an AnsibleAWSError"""
        if exception.exception:
            self.fail_json_aws(exception.exception, msg=exception.message)
        self.fail_json(msg=exception.message)

    def _gather_versions(self):
        """Gather AWS SDK (boto3 and botocore) dependency versions

        Returns {'boto3_version': str, 'botocore_version': str}
        Returns {} if either is not installed
        """
        return gather_sdk_versions()

    def require_boto3_at_least(self, desired, **kwargs):
        """Check if the available boto3 version is greater than or equal to a desired version.

        calls fail_json() when the boto3 version is less than the desired
        version

        Usage:
            module.require_boto3_at_least("1.2.3", reason="to update tags")
            module.require_boto3_at_least("1.1.1")

        :param desired the minimum desired version
        :param reason why the version is required (optional)
        """
        if not self.boto3_at_least(desired):
            self._module.fail_json(msg=missing_required_lib(f'boto3>={desired}', **kwargs), **self._gather_versions())

    def boto3_at_least(self, desired):
        return boto3_at_least(desired)

    def require_botocore_at_least(self, desired, **kwargs):
        """Check if the available botocore version is greater than or equal to a desired version.

        calls fail_json() when the botocore version is less than the desired
        version

        Usage:
            module.require_botocore_at_least("1.2.3", reason="to update tags")
            module.require_botocore_at_least("1.1.1")

        :param desired the minimum desired version
        :param reason why the version is required (optional)
        """
        if not self.botocore_at_least(desired):
            self._module.fail_json(msg=missing_required_lib(f'botocore>={desired}', **kwargs), **self._gather_versions())

    def botocore_at_least(self, desired):
        return botocore_at_least(desired)