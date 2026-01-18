from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def _get_firewall_rule_ids(self, name_or_id_list, filters=None):
    """
        Takes a list of firewall rule name or ids, looks them up and returns
        a list of firewall rule ids.

        Used by `create_firewall_policy` and `update_firewall_policy`.

        :param list[str] name_or_id_list: firewall rule name or id list
        :param dict filters: optional filters
        :raises: DuplicateResource on multiple matches
        :raises: NotFoundException if resource is not found
        :return: list of firewall rule ids
        :rtype: list[str]
        """
    if not filters:
        filters = {}
    ids_list = []
    for name_or_id in name_or_id_list:
        ids_list.append(self.network.find_firewall_rule(name_or_id, ignore_missing=False, **filters)['id'])
    return ids_list