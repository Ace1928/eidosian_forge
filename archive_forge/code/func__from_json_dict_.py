from typing import Any, Dict, List, Sequence, Type, Union
import cirq
@classmethod
def _from_json_dict_(cls, atol, eject_paulis, additional_gates, **kwargs):
    return cls(atol=atol, eject_paulis=eject_paulis, additional_gates=additional_gates)