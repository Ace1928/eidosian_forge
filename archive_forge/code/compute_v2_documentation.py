from keystoneauth1 import exceptions as ksa_exceptions
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.i18n import _
Delete a security group rule

        https://docs.openstack.org/api-ref/compute/#delete-security-group-rule

        :param string security_group_rule_id:
            Security group rule ID
        