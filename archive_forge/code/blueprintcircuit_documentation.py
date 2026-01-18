from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.parametertable import ParameterTable, ParameterView
Set the quantum registers associated with the circuit.