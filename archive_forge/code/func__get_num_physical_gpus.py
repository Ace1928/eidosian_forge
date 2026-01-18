import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def _get_num_physical_gpus():
    if RAY_ON_SPARK_WORKER_GPU_NUM in os.environ:
        return int(os.environ[RAY_ON_SPARK_WORKER_GPU_NUM])
    try:
        completed_proc = subprocess.run('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True, check=True, text=True, capture_output=True)
    except Exception as e:
        raise RuntimeError('Running command `nvidia-smi` for inferring GPU devices list failed.') from e
    return len(completed_proc.stdout.strip().split('\n'))