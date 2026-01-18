from typing import Union, Iterable, Tuple
from qiskit.pulse.instructions import Instruction
from qiskit.pulse.schedule import ScheduleBlock, Schedule
from qiskit.pulse.transforms import canonicalization
A helper function to convert instructions into list of instructions.