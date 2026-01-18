from __future__ import annotations
import inspect
import io
import re
from collections.abc import Iterator, Iterable, Callable
from functools import wraps
from typing import Union, List, Any
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager.passmanager import BasePassManager
from qiskit.passmanager.base_tasks import Task
from qiskit.passmanager.flow_controllers import FlowControllerLinear
from qiskit.passmanager.exceptions import PassManagerError
from .basepasses import BasePass
from .exceptions import TranspilerError
from .layout import TranspileLayout
def _legacy_style_callback(callback: Callable):

    def _wrapped_callable(task, passmanager_ir, property_set, running_time, count):
        callback(pass_=task, dag=passmanager_ir, time=running_time, property_set=property_set, count=count)
    return _wrapped_callable