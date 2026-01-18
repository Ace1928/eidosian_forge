import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def _get_cpu_cores():
    import multiprocessing
    if RAY_ON_SPARK_WORKER_CPU_CORES in os.environ:
        return int(os.environ[RAY_ON_SPARK_WORKER_CPU_CORES])
    return multiprocessing.cpu_count()