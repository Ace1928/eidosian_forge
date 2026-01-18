from __future__ import annotations
import typing
from abc import ABC
import numpy as np
from numpy import linalg as la
from .approximate import ApproximatingObjective
from .elementary_operations import ry_matrix, rz_matrix, place_unitary, place_cnot, rx_matrix

        Args:
            num_qubits: number of qubits.
            cnots: a CNOT structure to be used in the optimization procedure.
        