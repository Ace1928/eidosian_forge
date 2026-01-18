from __future__ import annotations
import enum
import warnings
from collections.abc import Sequence
from math import pi, erf
import numpy as np
from qiskit.circuit import Instruction as CircuitInst
from qiskit.circuit.library.standard_gates import RZXGate
from qiskit.exceptions import QiskitError
from qiskit.pulse import (
from qiskit.pulse import builder
from qiskit.pulse.filters import filter_instructions
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.transpiler.target import Target
from .base_builder import CalibrationBuilder
from .exceptions import CalibrationNotAvailable
class RZXCalibrationBuilder(CalibrationBuilder):
    """
    Creates calibrations for RZXGate(theta) by stretching and compressing
    Gaussian square pulses in the CX gate. This is done by retrieving (for a given pair of
    qubits) the CX schedule in the instruction schedule map of the backend defaults.
    The CX schedule must be an echoed cross-resonance gate optionally with rotary tones.
    The cross-resonance drive tones and rotary pulses must be Gaussian square pulses.
    The width of the Gaussian square pulse is adjusted so as to match the desired rotation angle.
    If the rotation angle is small such that the width disappears then the amplitude of the
    zero width Gaussian square pulse (i.e. a Gaussian) is reduced to reach the target rotation
    angle. Additional details can be found in https://arxiv.org/abs/2012.11660.
    """

    def __init__(self, instruction_schedule_map: InstructionScheduleMap=None, verbose: bool=True, target: Target=None):
        """
        Initializes a RZXGate calibration builder.

        Args:
            instruction_schedule_map: The :obj:`InstructionScheduleMap` object representing the
                default pulse calibrations for the target backend
            verbose: Set True to raise a user warning when RZX schedule cannot be built.
            target: The :class:`~.Target` representing the target backend, if both
                 ``instruction_schedule_map`` and this are specified then this argument will take
                 precedence and ``instruction_schedule_map`` will be ignored.

        Raises:
            QiskitError: Instruction schedule map is not provided.
        """
        super().__init__()
        self._inst_map = instruction_schedule_map
        self._verbose = verbose
        if target:
            self._inst_map = target.instruction_schedule_map()
        if self._inst_map is None:
            raise QiskitError('Calibrations can only be added to Pulse-enabled backends')

    def supported(self, node_op: CircuitInst, qubits: list) -> bool:
        """Determine if a given node supports the calibration.

        Args:
            node_op: Target instruction object.
            qubits: Integer qubit indices to check.

        Returns:
            Return ``True`` is calibration can be provided.
        """
        return isinstance(node_op, RZXGate) and ('cx' in self._inst_map.instructions or 'ecr' in self._inst_map.instructions)

    @staticmethod
    @builder.macro
    def rescale_cr_inst(instruction: Play, theta: float, sample_mult: int=16) -> int:
        """A builder macro to play stretched pulse.

        Args:
            instruction: The instruction from which to create a new shortened or lengthened pulse.
            theta: desired angle, pi/2 is assumed to be the angle that the pulse in the given
                play instruction implements.
            sample_mult: All pulses must be a multiple of sample_mult.

        Returns:
            Duration of stretched pulse.

        Raises:
            QiskitError: if rotation angle is not assigned.
        """
        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError('Target rotation angle is not assigned.') from ex
        params = instruction.pulse.parameters.copy()
        risefall_sigma_ratio = (params['duration'] - params['width']) / params['sigma']
        risefall_area = params['sigma'] * np.sqrt(2 * pi) * erf(risefall_sigma_ratio)
        full_area = params['width'] + risefall_area
        cal_angle = pi / 2
        target_area = abs(theta) / cal_angle * full_area
        new_width = target_area - risefall_area
        if new_width >= 0:
            width = new_width
            params['amp'] *= np.sign(theta)
        else:
            width = 0
            params['amp'] *= np.sign(theta) * target_area / risefall_area
        round_duration = round((width + risefall_sigma_ratio * params['sigma']) / sample_mult) * sample_mult
        params['duration'] = round_duration
        params['width'] = width
        stretched_pulse = GaussianSquare(**params)
        builder.play(stretched_pulse, instruction.channel)
        return round_duration

    def get_calibration(self, node_op: CircuitInst, qubits: list) -> Schedule | ScheduleBlock:
        """Builds the calibration schedule for the RZXGate(theta) with echos.

        Args:
            node_op: Instruction of the RZXGate(theta). I.e. params[0] is theta.
            qubits: List of qubits for which to get the schedules. The first qubit is
                the control and the second is the target.

        Returns:
            schedule: The calibration schedule for the RZXGate(theta).

        Raises:
            QiskitError: if rotation angle is not assigned.
            QiskitError: If the control and target qubits cannot be identified.
            CalibrationNotAvailable: RZX schedule cannot be built for input node.
        """
        theta = node_op.params[0]
        try:
            theta = float(theta)
        except TypeError as ex:
            raise QiskitError('Target rotation angle is not assigned.') from ex
        if np.isclose(theta, 0.0):
            return ScheduleBlock(name='rzx(0.000)')
        cal_type, cr_tones, comp_tones = _check_calibration_type(self._inst_map, qubits)
        if cal_type in [CRCalType.DIRECT_CX_FORWARD, CRCalType.DIRECT_CX_REVERSE]:
            if self._verbose:
                warnings.warn(f'CR instruction for qubits {qubits} is likely {cal_type.value} sequence. Pulse stretch for this calibration is not currently implemented. RZX schedule is not generated for this qubit pair.', UserWarning)
            raise CalibrationNotAvailable
        if cal_type in [CRCalType.ECR_CX_FORWARD, CRCalType.ECR_FORWARD]:
            xgate = self._inst_map.get('x', qubits[0])
            with builder.build(default_alignment='sequential', name='rzx(%.3f)' % theta) as rzx_theta_native:
                for cr_tone, comp_tone in zip(cr_tones, comp_tones):
                    with builder.align_left():
                        self.rescale_cr_inst(cr_tone, theta)
                        self.rescale_cr_inst(comp_tone, theta)
                    builder.call(xgate)
            return rzx_theta_native
        xgate = self._inst_map.get('x', qubits[1])
        szc = self._inst_map.get('rz', qubits[1], pi / 2)
        sxc = self._inst_map.get('sx', qubits[1])
        szt = self._inst_map.get('rz', qubits[0], pi / 2)
        sxt = self._inst_map.get('sx', qubits[0])
        with builder.build(name='hadamard') as hadamard:
            builder.call(szc, name='szc')
            builder.call(sxc, name='sxc')
            builder.call(szc, name='szc')
            builder.call(szt, name='szt')
            builder.call(sxt, name='sxt')
            builder.call(szt, name='szt')
        with builder.build(default_alignment='sequential', name='rzx(%.3f)' % theta) as rzx_theta_flip:
            builder.call(hadamard, name='hadamard')
            for cr_tone, comp_tone in zip(cr_tones, comp_tones):
                with builder.align_left():
                    self.rescale_cr_inst(cr_tone, theta)
                    self.rescale_cr_inst(comp_tone, theta)
                builder.call(xgate)
            builder.call(hadamard, name='hadamard')
        return rzx_theta_flip