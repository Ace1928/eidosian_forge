import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def _get_spark_worker_total_shared_memory():
    import shutil
    if RAY_ON_SPARK_WORKER_SHARED_MEMORY_BYTES in os.environ:
        return int(os.environ[RAY_ON_SPARK_WORKER_SHARED_MEMORY_BYTES])
    return shutil.disk_usage('/dev/shm').total