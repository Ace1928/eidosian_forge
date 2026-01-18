import datetime
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Union
import duet
from google.protobuf import any_pb2
import cirq
from cirq_google.engine import abstract_program, engine_client, util
from cirq_google.cloud import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.api import v2
from cirq_google.engine import engine_job
from cirq_google.serialization import circuit_serializer
Deletes the job and result, if any.