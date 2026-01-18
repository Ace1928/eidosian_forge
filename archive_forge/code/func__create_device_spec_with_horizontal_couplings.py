from dataclasses import dataclass
from typing import Dict, List, Tuple
import unittest.mock as mock
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.devices import grid_device
def _create_device_spec_with_horizontal_couplings():
    grid_qubits = [cirq.GridQubit(i, j) for i in range(GRID_HEIGHT) for j in range(2)]
    spec = v2.device_pb2.DeviceSpecification()
    spec.valid_qubits.extend([v2.qubit_to_proto_id(q) for q in grid_qubits])
    qubit_pairs = []
    grid_targets = spec.valid_targets.add()
    grid_targets.name = '2_qubit_targets'
    grid_targets.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    for row in range(int(GRID_HEIGHT / 2)):
        qubit_pairs.append((cirq.GridQubit(row, 0), cirq.GridQubit(row, 1)))
    for row in range(int(GRID_HEIGHT / 2), GRID_HEIGHT):
        qubit_pairs.append((cirq.GridQubit(row, 1), cirq.GridQubit(row, 0)))
    for pair in qubit_pairs:
        new_target = grid_targets.targets.add()
        new_target.ids.extend([v2.qubit_to_proto_id(q) for q in pair])
    gate_names = ['syc', 'sqrt_iswap', 'sqrt_iswap_inv', 'cz', 'phased_xz', 'virtual_zpow', 'physical_zpow', 'coupler_pulse', 'meas', 'wait']
    gate_durations = [(n, i * 1000) for i, n in enumerate(gate_names)]
    for gate_name, duration in sorted(gate_durations):
        gate = spec.valid_gates.add()
        getattr(gate, gate_name).SetInParent()
        gate.gate_duration_picos = duration
    expected_gateset = cirq.Gateset(cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ]), cirq.ops.phased_x_z_gate.PhasedXZGate, cirq.ops.common_gates.XPowGate, cirq.ops.common_gates.YPowGate, cirq.ops.phased_x_gate.PhasedXPowGate, cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]), cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]), cirq_google.experimental.ops.coupler_pulse.CouplerPulse, cirq.ops.measurement_gate.MeasurementGate, cirq.ops.wait_gate.WaitGate)
    base_duration = cirq.Duration(picos=1000)
    expected_gate_durations = {cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]): base_duration * 0, cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP]): base_duration * 1, cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]): base_duration * 2, cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ]): base_duration * 3, cirq.GateFamily(cirq.ops.phased_x_z_gate.PhasedXZGate): base_duration * 4, cirq.GateFamily(cirq.ops.common_gates.XPowGate): base_duration * 4, cirq.GateFamily(cirq.ops.common_gates.YPowGate): base_duration * 4, cirq.GateFamily(cirq.ops.phased_x_gate.PhasedXPowGate): base_duration * 4, cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]): base_duration * 5, cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]): base_duration * 6, cirq.GateFamily(cirq_google.experimental.ops.coupler_pulse.CouplerPulse): base_duration * 7, cirq.GateFamily(cirq.ops.measurement_gate.MeasurementGate): base_duration * 8, cirq.GateFamily(cirq.ops.wait_gate.WaitGate): base_duration * 9}
    expected_target_gatesets = (cirq_google.GoogleCZTargetGateset(additional_gates=[cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]), cirq.ops.common_gates.XPowGate, cirq.ops.common_gates.YPowGate, cirq.ops.phased_x_gate.PhasedXPowGate, cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]), cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]), cirq_google.experimental.ops.coupler_pulse.CouplerPulse, cirq.ops.wait_gate.WaitGate]), cirq_google.SycamoreTargetGateset(), cirq.SqrtIswapTargetGateset(additional_gates=[cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV]), cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ]), cirq.ops.common_gates.XPowGate, cirq.ops.common_gates.YPowGate, cirq.ops.phased_x_gate.PhasedXPowGate, cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]), cirq.GateFamily(cirq.ops.common_gates.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]), cirq_google.experimental.ops.coupler_pulse.CouplerPulse, cirq.ops.wait_gate.WaitGate]))
    return (_DeviceInfo(grid_qubits, qubit_pairs, expected_gateset, expected_gate_durations, expected_target_gatesets), spec)