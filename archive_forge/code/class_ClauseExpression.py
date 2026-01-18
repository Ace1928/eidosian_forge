import pyparsing as pp
import netaddr
from functools import reduce
from operator import and_, or_
from ovs.flow.decoders import (
class ClauseExpression(object):
    """ A clause expression represents a specific expression in the filter.

    A clause has the following form:
        [field] [operator] [value]

    Valid operators are:
        = (equality)
        != (inequality)
        < (arithmetic less-than)
        > (arithmetic more-than)
        ~= (__contains__)

    When evaluated, the clause finds what relevant part of the flow to use for
    evaluation, tries to translate the clause value to the relevant type and
    performs the clause operation.

    Attributes:
        field (str): The flow field used in the clause.
        operator (str): The flow operator used in the clause.
        value (str): The value to perform the comparison against.
    """
    operators = {}
    type_decoders = {int: decode_int, netaddr.IPAddress: IPMask, netaddr.EUI: EthMask, bool: bool}

    def __init__(self, tokens):
        self.field = tokens[0]
        self.value = ''
        self.operator = ''
        if len(tokens) > 1:
            self.operator = tokens[1]
            self.value = tokens[2]

    def __repr__(self):
        return '{}(field: {}, operator: {}, value: {})'.format(self.__class__.__name__, self.field, self.operator, self.value)

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

    def evaluate(self, flow):
        """Returns whether the clause is satisfied by the flow.

        Args:
            flow (Flow): the flow to evaluate.
        """
        result = self._find_keyval_to_evaluate(flow)
        if not result:
            return EvaluationResult(False)
        keyval, data = result
        if not self.value and (not self.operator):
            return EvaluationResult(True, keyval)
        if isinstance(data, Decoder):
            decoder = data.__class__
        else:
            decoder = self.type_decoders.get(data.__class__) or decode_default
        decoded_value = decoder(self.value)
        if self.operator == '=':
            return EvaluationResult(decoded_value == data, keyval)
        elif self.operator == '<':
            return EvaluationResult(data < decoded_value, keyval)
        elif self.operator == '>':
            return EvaluationResult(data > decoded_value, keyval)
        elif self.operator == '~=':
            return EvaluationResult(decoded_value in data, keyval)