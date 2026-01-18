from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
Delete an access rule.

        :param access_rule: the access rule to be deleted
        :type access_rule: str or
            :class:`keystoneclient.v3.access_rules.AccessRule`
        :param string user: User ID

        :returns: response object with 204 status
        :rtype: :class:`requests.models.Response`

        