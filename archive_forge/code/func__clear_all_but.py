import os
import hashlib
import pickle
import time
import shutil
import glob
from ..interfaces.base import BaseInterface
from ..pipeline.engine import Node
from ..pipeline.engine.utils import modify_paths
def _clear_all_but(self, runs, warn=True):
    """Remove all the runs apart from those given to the function
        input.
        """
    rm_all_but(self.base_dir, set(runs.keys()), warn=warn)
    for dir_name, job_names in list(runs.items()):
        rm_all_but(os.path.join(self.base_dir, dir_name), job_names, warn=warn)