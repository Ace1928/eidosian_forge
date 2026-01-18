import os
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.lib.io import file_io
def _is_temp_dir(dirpath, strategy):
    return dirpath.endswith(_get_base_dirpath(strategy))