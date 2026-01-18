import contextlib
import decimal
import gc
import numpy as np
import os
import random
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import time
import pytest
import pyarrow as pa
import pyarrow.fs
def _configure_s3_limited_user(s3_server, policy):
    """
    Attempts to use the mc command to configure the minio server
    with a special user limited:limited123 which does not have
    permission to create buckets.  This mirrors some real life S3
    configurations where users are given strict permissions.

    Arrow S3 operations should still work in such a configuration
    (e.g. see ARROW-13685)
    """
    if sys.platform == 'win32':
        pytest.skip('The mc command is not installed on Windows')
    try:
        _ensure_minio_component_version('mc', 2021)
        _ensure_minio_component_version('minio', 2021)
        tempdir = s3_server['tempdir']
        host, port, access_key, secret_key = s3_server['connection']
        address = '{}:{}'.format(host, port)
        mcdir = os.path.join(tempdir, 'mc')
        if os.path.exists(mcdir):
            shutil.rmtree(mcdir)
        os.mkdir(mcdir)
        policy_path = os.path.join(tempdir, 'limited-buckets-policy.json')
        with open(policy_path, mode='w') as policy_file:
            policy_file.write(policy)
        _wait_for_minio_startup(mcdir, address, access_key, secret_key)
        _run_mc_command(mcdir, 'admin', 'policy', 'add', 'myminio/', 'no-create-buckets', policy_path)
        _run_mc_command(mcdir, 'admin', 'user', 'add', 'myminio/', 'limited', 'limited123')
        _run_mc_command(mcdir, 'admin', 'policy', 'set', 'myminio', 'no-create-buckets', 'user=limited')
        _run_mc_command(mcdir, 'mb', 'myminio/existing-bucket', '--ignore-existing')
    except FileNotFoundError:
        pytest.skip('Configuring limited s3 user failed')