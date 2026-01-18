import json
import os
import tensorflow.compat.v2 as tf
Set multi-worker cluster spec in TF_CONFIG environment variable.

    Args:
      worker_hosts: comma-separated list of worker ip:port pairs.

    Returns:
      Number of workers in the cluster.
    