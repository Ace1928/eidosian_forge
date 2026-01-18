import pyparsing as pp
import netaddr
from functools import reduce
from operator import and_, or_
from ovs.flow.decoders import (
def _find_data_in_kv(self, kv_list):
    """Find a KeyValue for evaluation in a list of KeyValue.

        Args:
            kv_list (list[KeyValue]): list of KeyValue to look into.

        Returns:
            If found, tuple (kv, data) where kv is the KeyValue that matched
            and data is the data to be used for evaluation. None if not found.
        """
    key_parts = self.field.split('.')
    field = key_parts[0]
    kvs = [kv for kv in kv_list if kv.key == field]
    if not kvs:
        return None
    for kv in kvs:
        if kv.key == self.field:
            return (kv, kv.value)
        if len(key_parts) > 1:
            data = kv.value
            for subkey in key_parts[1:]:
                try:
                    data = data.get(subkey)
                except Exception:
                    data = None
                    break
                if not data:
                    break
            if data:
                return (kv, data)
    return None