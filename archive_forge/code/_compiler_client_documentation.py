from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor

        Generate a randomized benchmarking sequence.
        