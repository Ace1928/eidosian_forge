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
def _serialize_run_context(self, sweeps: 'cirq.Sweepable', repetitions: int) -> any_pb2.Any:
    if self.proto_version != ProtoVersion.V2:
        raise ValueError(f'invalid run context proto version: {self.proto_version}')
    return util.pack_any(v2.run_context_to_proto(sweeps, repetitions))