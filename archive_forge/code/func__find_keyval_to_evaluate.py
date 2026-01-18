import pyparsing as pp
import netaddr
from functools import reduce
from operator import and_, or_
from ovs.flow.decoders import (
def _find_keyval_to_evaluate(self, flow):
    """Finds the key-value and data to use for evaluation on a flow.

        Args:
            flow(Flow): The flow where the lookup is performed.

        Returns:
            If found, tuple (kv, data) where kv is the KeyValue that matched
            and data is the data to be used for evaluation. None if not found.

        """
    for section in flow.sections:
        data = self._find_data_in_kv(section.data)
        if data:
            return data
    return None