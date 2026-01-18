import datetime
import enum
import random
import string
from typing import Dict, List, Optional, Sequence, Set, TypeVar, Union, TYPE_CHECKING
import duet
import google.auth
from google.protobuf import any_pb2
import cirq
from cirq._compat import deprecated
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.cloud import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.serialization import CIRCUIT_SERIALIZER, Serializer
from cirq_google.serialization.arg_func_langs import arg_to_proto
def _make_random_id(prefix: str, length: int=16):
    random_digits = [random.choice(string.ascii_uppercase + string.digits) for _ in range(length)]
    suffix = ''.join(random_digits)
    suffix += datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    return f'{prefix}{suffix}'