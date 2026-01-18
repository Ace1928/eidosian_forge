import os
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.lib.io import file_io
def _get_base_dirpath(strategy):
    task_id = strategy.extended._task_id
    return 'workertemp_' + str(task_id)