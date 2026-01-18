import hashlib
import re
import warnings
from enum import Enum
from functools import singledispatchmethod
from typing import Union, List, Iterator, Optional
import numpy as np
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse import channels, instructions, library
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import QiskitError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import QobjMeasurementOption, PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.utils import MeasLevel
class InstructionToQobjConverter:
    """Converts Qiskit Pulse in-memory representation into Qobj data.

    This converter converts the Qiskit Pulse in-memory representation into
    the transfer layer format to submit the data from client to the server.

    The transfer layer format must be the text representation that coforms to
    the `OpenPulse specification<https://arxiv.org/abs/1809.03452>`__.
    Extention to the OpenPulse can be achieved by subclassing this this with
    extra methods corresponding to each augumented instruction. For example,

    .. code-block:: python

        class MyConverter(InstructionToQobjConverter):

            def _convert_NewInstruction(self, instruction, time_offset):
                command_dict = {
                    'name': 'new_inst',
                    't0': time_offset + instruction.start_time,
                    'param1': instruction.param1,
                    'param2': instruction.param2
                }
                return self._qobj_model(**command_dict)

    where ``NewInstruction`` must be a class name of Qiskit Pulse instruction.
    """

    def __init__(self, qobj_model: PulseQobjInstruction, **run_config):
        """Create new converter.

        Args:
             qobj_model: Transfer layer data schema.
             run_config: Run configuration.
        """
        self._qobj_model = qobj_model
        self._run_config = run_config

    def __call__(self, shift: int, instruction: Union[instructions.Instruction, List[instructions.Acquire]]) -> PulseQobjInstruction:
        """Convert Qiskit in-memory representation to Qobj instruction.

        Args:
            instruction: Instruction data in Qiskit Pulse.

        Returns:
            Qobj instruction data.

        Raises:
            QiskitError: When list of instruction is provided except for Acquire.
        """
        if isinstance(instruction, list):
            if all((isinstance(inst, instructions.Acquire) for inst in instruction)):
                return self._convert_bundled_acquire(instruction_bundle=instruction, time_offset=shift)
            raise QiskitError('Bundle of instruction is not supported except for Acquire.')
        return self._convert_instruction(instruction, shift)

    @singledispatchmethod
    def _convert_instruction(self, instruction, time_offset: int) -> PulseQobjInstruction:
        raise QiskitError(f"Pulse Qobj doesn't support {instruction.__class__.__name__}. This instruction cannot be submitted with Qobj.")

    @_convert_instruction.register(instructions.Acquire)
    def _convert_acquire(self, instruction, time_offset: int) -> PulseQobjInstruction:
        """Return converted `Acquire`.

        Args:
            instruction: Qiskit Pulse acquire instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        meas_level = self._run_config.get('meas_level', 2)
        mem_slot = []
        if instruction.mem_slot:
            mem_slot = [instruction.mem_slot.index]
        command_dict = {'name': 'acquire', 't0': time_offset + instruction.start_time, 'duration': instruction.duration, 'qubits': [instruction.channel.index], 'memory_slot': mem_slot}
        if meas_level == MeasLevel.CLASSIFIED:
            if instruction.discriminator:
                command_dict.update({'discriminators': [QobjMeasurementOption(name=instruction.discriminator.name, params=instruction.discriminator.params)]})
            if instruction.reg_slot:
                command_dict.update({'register_slot': [instruction.reg_slot.index]})
        if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
            if instruction.kernel:
                command_dict.update({'kernels': [QobjMeasurementOption(name=instruction.kernel.name, params=instruction.kernel.params)]})
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.SetFrequency)
    def _convert_set_frequency(self, instruction, time_offset: int) -> PulseQobjInstruction:
        """Return converted `SetFrequency`.

        Args:
            instruction: Qiskit Pulse set frequency instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {'name': 'setf', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'frequency': instruction.frequency / 1000000000.0}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.ShiftFrequency)
    def _convert_shift_frequency(self, instruction, time_offset: int) -> PulseQobjInstruction:
        """Return converted `ShiftFrequency`.

        Args:
            instruction: Qiskit Pulse shift frequency instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {'name': 'shiftf', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'frequency': instruction.frequency / 1000000000.0}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.SetPhase)
    def _convert_set_phase(self, instruction, time_offset: int) -> PulseQobjInstruction:
        """Return converted `SetPhase`.

        Args:
            instruction: Qiskit Pulse set phase instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {'name': 'setp', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'phase': instruction.phase}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.ShiftPhase)
    def _convert_shift_phase(self, instruction, time_offset: int) -> PulseQobjInstruction:
        """Return converted `ShiftPhase`.

        Args:
            instruction: Qiskit Pulse shift phase instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {'name': 'fc', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'phase': instruction.phase}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.Delay)
    def _convert_delay(self, instruction, time_offset: int) -> PulseQobjInstruction:
        """Return converted `Delay`.

        Args:
            instruction: Qiskit Pulse delay instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        command_dict = {'name': 'delay', 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'duration': instruction.duration}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.Play)
    def _convert_play(self, instruction, time_offset: int) -> PulseQobjInstruction:
        """Return converted `Play`.

        Args:
            instruction: Qiskit Pulse play instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.
        """
        if isinstance(instruction.pulse, library.SymbolicPulse):
            params = dict(instruction.pulse.parameters)
            if 'amp' in params and 'angle' in params:
                params['amp'] = complex(params['amp'] * np.exp(1j * params['angle']))
                del params['angle']
            command_dict = {'name': 'parametric_pulse', 'pulse_shape': ParametricPulseShapes.from_instance(instruction.pulse).name, 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name, 'parameters': params}
        else:
            command_dict = {'name': instruction.name, 't0': time_offset + instruction.start_time, 'ch': instruction.channel.name}
        return self._qobj_model(**command_dict)

    @_convert_instruction.register(instructions.Snapshot)
    def _convert_snapshot(self, instruction, time_offset: int) -> PulseQobjInstruction:
        """Return converted `Snapshot`.

        Args:
            time_offset: Offset time.
            instruction: Qiskit Pulse snapshot instruction.

        Returns:
            Qobj instruction data.
        """
        command_dict = {'name': 'snapshot', 't0': time_offset + instruction.start_time, 'label': instruction.label, 'type': instruction.type}
        return self._qobj_model(**command_dict)

    def _convert_bundled_acquire(self, instruction_bundle: List[instructions.Acquire], time_offset: int) -> PulseQobjInstruction:
        """Return converted list of parallel `Acquire` instructions.

        Args:
            instruction_bundle: List of Qiskit Pulse acquire instruction.
            time_offset: Offset time.

        Returns:
            Qobj instruction data.

        Raises:
            QiskitError: When instructions are not aligned.
            QiskitError: When instructions have different duration.
            QiskitError: When discriminator or kernel is missing in a part of instructions.
        """
        meas_level = self._run_config.get('meas_level', 2)
        t0 = instruction_bundle[0].start_time
        duration = instruction_bundle[0].duration
        memory_slots = []
        register_slots = []
        qubits = []
        discriminators = []
        kernels = []
        for instruction in instruction_bundle:
            qubits.append(instruction.channel.index)
            if instruction.start_time != t0:
                raise QiskitError('The supplied acquire instructions have different starting times. Something has gone wrong calling this code. Please report this issue.')
            if instruction.duration != duration:
                raise QiskitError('Acquire instructions beginning at the same time must have same duration.')
            if instruction.mem_slot:
                memory_slots.append(instruction.mem_slot.index)
            if meas_level == MeasLevel.CLASSIFIED:
                if instruction.discriminator:
                    discriminators.append(QobjMeasurementOption(name=instruction.discriminator.name, params=instruction.discriminator.params))
                if instruction.reg_slot:
                    register_slots.append(instruction.reg_slot.index)
            if meas_level in [MeasLevel.KERNELED, MeasLevel.CLASSIFIED]:
                if instruction.kernel:
                    kernels.append(QobjMeasurementOption(name=instruction.kernel.name, params=instruction.kernel.params))
        command_dict = {'name': 'acquire', 't0': time_offset + t0, 'duration': duration, 'qubits': qubits}
        if memory_slots:
            command_dict['memory_slot'] = memory_slots
        if register_slots:
            command_dict['register_slot'] = register_slots
        if discriminators:
            num_discriminators = len(discriminators)
            if num_discriminators == len(qubits) or num_discriminators == 1:
                command_dict['discriminators'] = discriminators
            else:
                raise QiskitError('A discriminator must be supplied for every acquisition or a single discriminator for all acquisitions.')
        if kernels:
            num_kernels = len(kernels)
            if num_kernels == len(qubits) or num_kernels == 1:
                command_dict['kernels'] = kernels
            else:
                raise QiskitError('A kernel must be supplied for every acquisition or a single kernel for all acquisitions.')
        return self._qobj_model(**command_dict)