import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info
def _chunk_dict(d, chunk_size):
    """
    Splits a dictionary into chunks of the specified size.
    Taken from: https://stackoverflow.com/a/22878842
    """
    it = iter(d)
    for _ in range(0, len(d), chunk_size):
        yield {k: d[k] for k in islice(it, chunk_size)}