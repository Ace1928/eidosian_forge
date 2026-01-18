from keystoneclient import base
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import policies
List endpoints with the effective association to a policy.

        :param policy: policy object or ID

        :returns: list of endpoints that are associated with the policy

        