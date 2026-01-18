import datetime
import functools
from typing import Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from cirq.protocols.json_serialization import ObjectFactory
def _symmetricalqidpair(qids):
    return frozenset(qids)