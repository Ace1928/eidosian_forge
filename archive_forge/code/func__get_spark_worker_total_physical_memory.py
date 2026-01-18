import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def _get_spark_worker_total_physical_memory():
    import psutil
    if RAY_ON_SPARK_WORKER_PHYSICAL_MEMORY_BYTES in os.environ:
        return int(os.environ[RAY_ON_SPARK_WORKER_PHYSICAL_MEMORY_BYTES])
    return psutil.virtual_memory().total