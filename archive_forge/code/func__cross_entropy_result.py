import datetime
import functools
from typing import Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from cirq.protocols.json_serialization import ObjectFactory
def _cross_entropy_result(data, repetitions, **kwargs) -> CrossEntropyResult:
    purity_data = kwargs.get('purity_data', None)
    if purity_data is not None:
        purity_data = [SpecklePurityPair(d, f) for d, f in purity_data]
    return CrossEntropyResult(data=[CrossEntropyPair(d, f) for d, f in data], repetitions=repetitions, purity_data=purity_data)