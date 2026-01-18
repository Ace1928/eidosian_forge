import copy
import dataclasses
import logging
import functools
import time
import numpy as np
import rustworkx as rx
from qiskit.converters import dag_to_circuit
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.sabre_layout import sabre_layout_and_routing
from qiskit._accelerate.sabre_swap import (
from qiskit.transpiler.passes.routing.sabre_swap import _build_sabre_dag, _apply_sabre_result
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.utils.parallel import CPU_COUNT
def _ancilla_allocation_no_pass_manager(self, dag):
    """Run the ancilla-allocation and -enlargment passes on the DAG chained onto our
        ``property_set``, skipping the DAG-to-circuit conversion cost of using a ``PassManager``."""
    ancilla_pass = FullAncillaAllocation(self.coupling_map)
    ancilla_pass.property_set = self.property_set
    dag = ancilla_pass.run(dag)
    enlarge_pass = EnlargeWithAncilla()
    enlarge_pass.property_set = ancilla_pass.property_set
    dag = enlarge_pass.run(dag)
    self.property_set = enlarge_pass.property_set
    return dag