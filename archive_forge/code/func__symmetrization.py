import itertools
import warnings
from math import log, pi
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._compiler import AbstractCompiler, QVMCompiler
from pyquil.api._qam import QAM
from pyquil.api._qpu import QPU
from pyquil.api._quantum_computer import QuantumComputer as QuantumComputerV3
from pyquil.api._quantum_computer import get_qc as get_qc_v3, QuantumExecutable
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.gates import MEASURE, RX
from pyquil.noise import NoiseModel, decoherence_noise_with_asymmetric_ro
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import AbstractQuantumProcessor, NxQuantumProcessor
from pyquil.quil import Program, validate_supported_quil
from pyquil.quilatom import qubit_index
from ._qam import StatefulQAM
def _symmetrization(program: Program, meas_qubits: List[int], symm_type: int=3) -> Tuple[List[Program], List[Tuple[bool]]]:
    """
    For the input program generate new programs which flip the measured qubits with an X gate in
    certain combinations in order to symmetrize readout.

    An expanded list of programs is returned along with a list of bools which indicates which
    qubits are flipped in each program.

    The symmetrization types are specified by an int; the types available are:

    * -1 -- exhaustive symmetrization uses every possible combination of flips
    *  0 -- trivial that is no symmetrization
    *  1 -- symmetrization using an OA with strength 1
    *  2 -- symmetrization using an OA with strength 2
    *  3 -- symmetrization using an OA with strength 3

    In the context of readout symmetrization the strength of the orthogonal array enforces the
    symmetry of the marginal confusion matrices.

    By default a strength 3 OA is used; this ensures expectations of the form <b_k * b_j * b_i>
    for bits any bits i,j,k will have symmetric readout errors. Here expectation of a random
    variable x as is denote <x> = sum_i Pr(i) x_i. It turns out that a strength 3 OA is also a
    strength 2 and strength 1 OA it also ensures <b_j * b_i> and <b_i> have symmetric readout
    errors for any bits b_j and b_i.

    :param programs: a program which will be symmetrized.
    :param meas_qubits: the groups of measurement qubits. Only these qubits will be symmetrized
        over, even if the program acts on other qubits.
    :param sym_type: an int determining the type of symmetrization performed.
    :return: a list of symmetrized programs, the corresponding array of bools indicating which
        qubits were flipped.
    """
    if symm_type < -1 or symm_type > 3:
        raise ValueError('symm_type must be one of the following ints [-1, 0, 1, 2, 3].')
    elif symm_type == -1:
        flip_matrix = np.asarray(list(itertools.product([0, 1], repeat=len(meas_qubits))))
    elif symm_type >= 0:
        flip_matrix = _construct_orthogonal_array(len(meas_qubits), symm_type)
    flip_matrix = flip_matrix[:, :len(meas_qubits)]
    symm_programs = []
    flip_arrays = []
    for flip_array in flip_matrix:
        total_prog_symm = program.copy()
        prog_symm = _flip_array_to_prog(flip_array, meas_qubits)
        total_prog_symm += prog_symm
        symm_programs.append(total_prog_symm)
        flip_arrays.append(flip_array)
    return (symm_programs, flip_arrays)