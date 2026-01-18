import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def _resolve_hostlist(self):
    """Returns a list of hostnames for nodes running the current job step."""
    return expand_hostlist(_get_slurm_var('STEP_NODELIST'))