import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def _build_create_or_update_params(self, stack_name, template_body, template_url, parameters, disable_rollback, timeout_in_minutes, notification_arns, capabilities, on_failure, stack_policy_body, stack_policy_url, tags, use_previous_template=None, stack_policy_during_update_body=None, stack_policy_during_update_url=None):
    """
        Helper that creates JSON parameters needed by a Stack Create or
        Stack Update call.

        :type stack_name: string
        :param stack_name:
        The name associated with the stack. The name must be unique within your
            AWS account.

        Must contain only alphanumeric characters (case sensitive) and start
            with an alpha character. Maximum length of the name is 255
            characters.

        :type template_body: string
        :param template_body: Structure containing the template body. (For more
            information, go to `Template Anatomy`_ in the AWS CloudFormation
            User Guide.)
        Conditional: You must pass either `UsePreviousTemplate` or one of
            `TemplateBody` or `TemplateUrl`. If both `TemplateBody` and
            `TemplateUrl` are passed, only `TemplateBody` is used.
            `TemplateBody`.

        :type template_url: string
        :param template_url: Location of file containing the template body. The
            URL must point to a template (max size: 307,200 bytes) located in
            an S3 bucket in the same region as the stack. For more information,
            go to the `Template Anatomy`_ in the AWS CloudFormation User Guide.
        Conditional: You must pass either `UsePreviousTemplate` or one of
            `TemplateBody` or `TemplateUrl`. If both `TemplateBody` and
            `TemplateUrl` are passed, only `TemplateBody` is used.
            `TemplateBody`.

        :type parameters: list
        :param parameters: A list of key/value tuples that specify input
            parameters for the stack. A 3-tuple (key, value, bool) may be used to
            specify the `UsePreviousValue` option.

        :type disable_rollback: boolean
        :param disable_rollback: Set to `True` to disable rollback of the stack
            if stack creation failed. You can specify either `DisableRollback`
            or `OnFailure`, but not both.
        Default: `False`

        :type timeout_in_minutes: integer
        :param timeout_in_minutes: The amount of time that can pass before the
            stack status becomes CREATE_FAILED; if `DisableRollback` is not set
            or is set to `False`, the stack will be rolled back.

        :type notification_arns: list
        :param notification_arns: The Simple Notification Service (SNS) topic
            ARNs to publish stack related events. You can find your SNS topic
            ARNs using the `SNS console`_ or your Command Line Interface (CLI).

        :type capabilities: list
        :param capabilities: The list of capabilities that you want to allow in
            the stack. If your template contains certain resources, you must
            specify the CAPABILITY_IAM value for this parameter; otherwise,
            this action returns an InsufficientCapabilities error. The
            following resources require you to specify the capabilities
            parameter: `AWS::CloudFormation::Stack`_, `AWS::IAM::AccessKey`_,
            `AWS::IAM::Group`_, `AWS::IAM::InstanceProfile`_,
            `AWS::IAM::Policy`_, `AWS::IAM::Role`_, `AWS::IAM::User`_, and
            `AWS::IAM::UserToGroupAddition`_.

        :type on_failure: string
        :param on_failure: Determines what action will be taken if stack
            creation fails. This must be one of: DO_NOTHING, ROLLBACK, or
            DELETE. You can specify either `OnFailure` or `DisableRollback`,
            but not both.
        Default: `ROLLBACK`

        :type stack_policy_body: string
        :param stack_policy_body: Structure containing the stack policy body.
            (For more information, go to ` Prevent Updates to Stack Resources`_
            in the AWS CloudFormation User Guide.)
        If you pass `StackPolicyBody` and `StackPolicyURL`, only
            `StackPolicyBody` is used.

        :type stack_policy_url: string
        :param stack_policy_url: Location of a file containing the stack
            policy. The URL must point to a policy (max size: 16KB) located in
            an S3 bucket in the same region as the stack. If you pass
            `StackPolicyBody` and `StackPolicyURL`, only `StackPolicyBody` is
            used.

        :type tags: list
        :param tags: A set of user-defined `Tags` to associate with this stack,
            represented by key/value pairs. Tags defined for the stack are
            propagated to EC2 resources that are created as part of the stack.
            A maximum number of 10 tags can be specified.

        :type use_previous_template: boolean
        :param use_previous_template: Set to `True` to use the previous
            template instead of uploading a new one via `TemplateBody` or
            `TemplateURL`.
        Conditional: You must pass either `UsePreviousTemplate` or one of
            `TemplateBody` or `TemplateUrl`.

        :type stack_policy_during_update_body: string
        :param stack_policy_during_update_body: Structure containing the
            temporary overriding stack policy body. If you pass
            `StackPolicyDuringUpdateBody` and `StackPolicyDuringUpdateURL`,
            only `StackPolicyDuringUpdateBody` is used.
        If you want to update protected resources, specify a temporary
            overriding stack policy during this update. If you do not specify a
            stack policy, the current policy that associated with the stack
            will be used.

        :type stack_policy_during_update_url: string
        :param stack_policy_during_update_url: Location of a file containing
            the temporary overriding stack policy. The URL must point to a
            policy (max size: 16KB) located in an S3 bucket in the same region
            as the stack. If you pass `StackPolicyDuringUpdateBody` and
            `StackPolicyDuringUpdateURL`, only `StackPolicyDuringUpdateBody` is
            used.
        If you want to update protected resources, specify a temporary
            overriding stack policy during this update. If you do not specify a
            stack policy, the current policy that is associated with the stack
            will be used.

        :rtype: dict
        :return: JSON parameters represented as a Python dict.
        """
    params = {'ContentType': 'JSON', 'StackName': stack_name, 'DisableRollback': self.encode_bool(disable_rollback)}
    if template_body:
        params['TemplateBody'] = template_body
    if template_url:
        params['TemplateURL'] = template_url
    if use_previous_template is not None:
        params['UsePreviousTemplate'] = self.encode_bool(use_previous_template)
    if template_body and template_url:
        boto.log.warning('If both TemplateBody and TemplateURL are specified, only TemplateBody will be honored by the API')
    if parameters and len(parameters) > 0:
        for i, parameter_tuple in enumerate(parameters):
            key, value = parameter_tuple[:2]
            use_previous = parameter_tuple[2] if len(parameter_tuple) > 2 else False
            params['Parameters.member.%d.ParameterKey' % (i + 1)] = key
            if use_previous:
                params['Parameters.member.%d.UsePreviousValue' % (i + 1)] = self.encode_bool(use_previous)
            else:
                params['Parameters.member.%d.ParameterValue' % (i + 1)] = value
    if capabilities:
        for i, value in enumerate(capabilities):
            params['Capabilities.member.%d' % (i + 1)] = value
    if tags:
        for i, (key, value) in enumerate(tags.items()):
            params['Tags.member.%d.Key' % (i + 1)] = key
            params['Tags.member.%d.Value' % (i + 1)] = value
    if notification_arns and len(notification_arns) > 0:
        self.build_list_params(params, notification_arns, 'NotificationARNs.member')
    if timeout_in_minutes:
        params['TimeoutInMinutes'] = int(timeout_in_minutes)
    if disable_rollback is not None:
        params['DisableRollback'] = str(disable_rollback).lower()
    if on_failure is not None:
        params['OnFailure'] = on_failure
    if stack_policy_body is not None:
        params['StackPolicyBody'] = stack_policy_body
    if stack_policy_url is not None:
        params['StackPolicyURL'] = stack_policy_url
    if stack_policy_during_update_body is not None:
        params['StackPolicyDuringUpdateBody'] = stack_policy_during_update_body
    if stack_policy_during_update_url is not None:
        params['StackPolicyDuringUpdateURL'] = stack_policy_during_update_url
    return params