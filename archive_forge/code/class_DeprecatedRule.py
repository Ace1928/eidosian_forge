import collections.abc
import copy
import logging
import os
import typing as ty
import warnings
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import strutils
import yaml
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_policy import _parser
from oslo_policy import opts
class DeprecatedRule(_BaseRule):
    """Represents a Deprecated policy or rule.

    Here's how you can use it to change a policy's default role or rule. Assume
    the following policy exists in code::

        from oslo_policy import policy

        policy.DocumentedRuleDefault(
            name='foo:create_bar',
            check_str='role:fizz',
            description='Create a bar.',
            operations=[{'path': '/v1/bars', 'method': 'POST'}]
        )

    The next snippet will maintain the deprecated option, but allow
    ``foo:create_bar`` to default to ``role:bang`` instead of ``role:fizz``::

        deprecated_rule = policy.DeprecatedRule(
            name='foo:create_bar',
            check_str='role:fizz'
            deprecated_reason='role:bang is a better default',
            deprecated_since='N',
        )

        policy.DocumentedRuleDefault(
            name='foo:create_bar',
            check_str='role:bang',
            description='Create a bar.',
            operations=[{'path': '/v1/bars', 'method': 'POST'}],
            deprecated_rule=deprecated_rule,
        )

    DeprecatedRule can be used to change the policy name itself. Assume the
    following policy exists in code::

        from oslo_policy import policy

        policy.DocumentedRuleDefault(
            name='foo:post_bar',
            check_str='role:fizz',
            description='Create a bar.',
            operations=[{'path': '/v1/bars', 'method': 'POST'}]
        )

    For the sake of consistency, let's say we want to replace ``foo:post_bar``
    with ``foo:create_bar``, but keep the same ``check_str`` as the default. We
    can accomplish this by doing::

        deprecated_rule = policy.DeprecatedRule(
            name='foo:post_bar',
            check_str='role:fizz'
            deprecated_reason='foo:create_bar is more consistent',
            deprecated_since='N',
        )

        policy.DocumentedRuleDefault(
            name='foo:create_bar',
            check_str='role:fizz',
            description='Create a bar.',
            operations=[{'path': '/v1/bars', 'method': 'POST'}],
            deprecated_rule=deprecated_rule,
        )

    Finally, let's use DeprecatedRule to break a policy into more granular
    policies. Let's assume the following policy exists in code::

        policy.DocumentedRuleDefault(
            name='foo:bar',
            check_str='role:bazz',
            description='Create, read, update, or delete a bar.',
            operations=[
                {
                    'path': '/v1/bars',
                    'method': 'POST'
                },
                {
                    'path': '/v1/bars',
                    'method': 'GET'
                },
                {
                    'path': '/v1/bars/{bar_id}',
                    'method': 'GET'
                },
                {
                    'path': '/v1/bars/{bar_id}',
                    'method': 'PATCH'
                },
                {
                    'path': '/v1/bars/{bar_id}',
                    'method': 'DELETE'
                }
            ]
        )

    Here we can see the same policy is used to protect multiple operations on
    bars. This prevents operators from being able to assign different roles to
    different actions that can be taken on bar. For example, what if an
    operator wanted to require a less restrictive role or rule to list bars but
    a more restrictive rule to delete them? The following will introduce a
    policy that helps achieve that and deprecate the original, overly-broad
    policy::

        deprecated_rule = policy.DeprecatedRule(
            name='foo:bar',
            check_str='role:bazz'
            deprecated_reason=(
                'foo:bar has been replaced by more granular policies'
            ),
            deprecated_since='N',
        )

        policy.DocumentedRuleDefault(
            name='foo:create_bar',
            check_str='role:bang',
            description='Create a bar.',
            operations=[{'path': '/v1/bars', 'method': 'POST'}],
            deprecated_rule=deprecated_rule,
        )
        policy.DocumentedRuleDefault(
            name='foo:list_bars',
            check_str='role:bazz',
            description='List bars.',
            operations=[{'path': '/v1/bars', 'method': 'GET'}],
            deprecated_rule=deprecated_rule,
        )
        policy.DocumentedRuleDefault(
            name='foo:get_bar',
            check_str='role:bazz',
            description='Get a bar.',
            operations=[{'path': '/v1/bars/{bar_id}', 'method': 'GET'}],
            deprecated_rule=deprecated_rule,
        )
        policy.DocumentedRuleDefault(
            name='foo:update_bar',
            check_str='role:bang',
            description='Update a bar.',
            operations=[{'path': '/v1/bars/{bar_id}', 'method': 'PATCH'}],
            deprecated_rule=deprecated_rule,
        )
        policy.DocumentedRuleDefault(
            name='foo:delete_bar',
            check_str='role:bang',
            description='Delete a bar.',
            operations=[{'path': '/v1/bars/{bar_id}', 'method': 'DELETE'}],
            deprecated_rule=deprecated_rule,
        )

    :param name: The name of the policy. This is used when referencing it
        from another rule or during policy enforcement.
    :param check_str: The policy. This is a string  defining a policy that
        conforms to the policy language outlined at the top of the file.
    :param deprecated_reason: indicates why this policy is planned for removal
        in a future release.
    :param deprecated_since: indicates which release this policy was deprecated
        in. Accepts any string, though valid version strings are encouraged.

    .. versionchanged:: 1.29
       Added *DeprecatedRule* object.

    .. versionchanged:: 3.4
       Added *deprecated_reason* parameter.

    .. versionchanged:: 3.4
       Added *deprecated_since* parameter.
    """

    def __init__(self, name: str, check_str: str, *, deprecated_reason: ty.Optional[str]=None, deprecated_since: ty.Optional[str]=None):
        super().__init__(name, check_str)
        self._deprecated_reason = deprecated_reason
        self._deprecated_since = deprecated_since
        if not deprecated_reason or not deprecated_since:
            warnings.warn(f'{name} deprecated without deprecated_reason or deprecated_since. This will be an error in a future release', DeprecationWarning)

    @property
    def deprecated_reason(self):
        return self._deprecated_reason

    @property
    def deprecated_since(self):
        return self._deprecated_since