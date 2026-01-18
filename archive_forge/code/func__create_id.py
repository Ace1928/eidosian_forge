import abc
import datetime
from typing import Dict, List, Optional, overload, TYPE_CHECKING, Union
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine import calibration
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_processor import AbstractProcessor
from cirq_google.engine.abstract_program import AbstractProgram
def _create_id(self, id_type: str='reservation') -> str:
    """Creates a unique resource id for child objects."""
    self._resource_id_counter += 1
    return f'projects/{self._project_name}/processors/{self._processor_id}/{id_type}/{self._resource_id_counter}'