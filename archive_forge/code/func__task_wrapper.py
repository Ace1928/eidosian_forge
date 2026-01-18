from the multiprocessing library.
import os
from concurrent.futures import ProcessPoolExecutor
import sys
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import local_hardware_info
from qiskit import user_config
def _task_wrapper(param):
    task, value, task_args, task_kwargs = param
    return task(value, *task_args, **task_kwargs)