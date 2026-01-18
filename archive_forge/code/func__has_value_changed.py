from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
def _has_value_changed(consul_client, key, target_value):
    """
    Uses the given Consul client to determine if the value associated to the given key is different to the given target
    value.
    :param consul_client: Consul connected client
    :param key: key in Consul
    :param target_value: value to be associated to the key
    :return: tuple where the first element is the value of the "X-Consul-Index" header and the second is `True` if the
    value has changed (i.e. the stored value is not the target value)
    """
    index, existing = consul_client.kv.get(key)
    if not existing:
        return (index, True)
    try:
        changed = to_text(existing['Value'], errors='surrogate_or_strict') != target_value
        return (index, changed)
    except UnicodeError:
        return (index, True)