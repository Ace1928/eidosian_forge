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
class RuleDefault(_BaseRule):
    """A class for holding policy definitions.

    It is required to supply a name and value at creation time. It is
    encouraged to also supply a description to assist operators.

    :param name: The name of the policy. This is used when referencing it
        from another rule or during policy enforcement.
    :param check_str: The policy. This is a string  defining a policy that
        conforms to the policy language outlined at the top of the file.
    :param description: A plain text description of the policy. This will be
        used to comment sample policy files for use by deployers.
    :param deprecated_rule: :class:`.DeprecatedRule`
    :param deprecated_for_removal: indicates whether the policy is planned for
        removal in a future release.
    :param deprecated_reason: indicates why this policy is planned for removal
        in a future release. Silently ignored if deprecated_for_removal is
        False.
    :param deprecated_since: indicates which release this policy was deprecated
        in. Accepts any string, though valid version strings are encouraged.
        Silently ignored if deprecated_for_removal is False.
    :param scope_types: A list containing the intended scopes of the operation
        being done.

    .. versionchanged:: 1.29
       Added *deprecated_rule* parameter.

    .. versionchanged:: 1.29
       Added *deprecated_for_removal* parameter.

    .. versionchanged:: 1.29
       Added *deprecated_reason* parameter.

    .. versionchanged:: 1.29
       Added *deprecated_since* parameter.

    .. versionchanged:: 1.31
       Added *scope_types* parameter.
    """

    def __init__(self, name, check_str, description=None, deprecated_rule=None, deprecated_for_removal=False, deprecated_reason=None, deprecated_since=None, scope_types=None):
        super().__init__(name, check_str)
        self._description = description
        self._deprecated_rule = copy.deepcopy(deprecated_rule) or []
        self._deprecated_for_removal = deprecated_for_removal
        self._deprecated_reason = deprecated_reason
        self._deprecated_since = deprecated_since
        if self.deprecated_rule:
            if not isinstance(self.deprecated_rule, DeprecatedRule):
                raise ValueError('deprecated_rule must be a DeprecatedRule object.')
        if deprecated_for_removal:
            if deprecated_reason is None or deprecated_since is None:
                raise ValueError('%(name)s deprecated without deprecated_reason or deprecated_since. Both must be supplied if deprecating a policy' % {'name': self.name})
        elif deprecated_rule and (deprecated_reason or deprecated_since):
            warnings.warn(f'{name} should not configure deprecated_reason or deprecated_since as these should be configured on the DeprecatedRule indicated by deprecated_rule. This will be an error in a future release', DeprecationWarning)
        if scope_types:
            msg = 'scope_types must be a list of strings.'
            if not isinstance(scope_types, list):
                raise ValueError(msg)
            for scope_type in scope_types:
                if not isinstance(scope_type, str):
                    raise ValueError(msg)
                if scope_types.count(scope_type) > 1:
                    raise ValueError('scope_types must be a list of unique strings.')
        self.scope_types = scope_types

    @property
    def description(self):
        return self._description

    @property
    def deprecated_rule(self):
        return self._deprecated_rule

    @property
    def deprecated_for_removal(self):
        return self._deprecated_for_removal

    @property
    def deprecated_reason(self):
        return self._deprecated_reason

    @property
    def deprecated_since(self):
        return self._deprecated_since

    def __eq__(self, other):
        """Equality operator.

        All check objects have a stable string representation. It is used for
        comparison rather than check_str because multiple check_str's may parse
        to the same check object. For instance '' and '@' are equivalent and
        the parsed rule string representation for both is '@'.

        The description does not play a role in the meaning of the check so it
        is not considered for equality.
        """
        if self.name == other.name and str(self.check) == str(other.check) and (isinstance(self, other.__class__) or isinstance(other, self.__class__)):
            return True
        return False