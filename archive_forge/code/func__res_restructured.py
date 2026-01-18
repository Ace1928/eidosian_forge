import inspect
import logging
import warnings
import tensorflow as tf
from tensorflow.python.eager import context
import pennylane as qml
from pennylane.measurements import Shots
def _res_restructured(res, tapes):
    """
    Reconstruct the nested tuple structure of the output of a list of tapes
    """
    start = 0
    res_nested = []
    for tape in tapes:
        tape_shots = tape.shots or Shots(1)
        shot_res_nested = []
        num_meas = len(tape.measurements)
        for _ in range(tape_shots.num_copies):
            shot_res = tuple(res[start:start + num_meas])
            shot_res_nested.append(shot_res[0] if num_meas == 1 else shot_res)
            start += num_meas
        res_nested.append(tuple(shot_res_nested) if tape_shots.has_partitioned_shots else shot_res_nested[0])
    return tuple(res_nested)