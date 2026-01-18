import hashlib
import logging
import sys
from optparse import Values
from typing import List
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.utils.hashes import FAVORITE_HASH, STRONG_HASHES
from pip._internal.utils.misc import read_chunks, write_output
def _hash_of_file(path: str, algorithm: str) -> str:
    """Return the hash digest of a file."""
    with open(path, 'rb') as archive:
        hash = hashlib.new(algorithm)
        for chunk in read_chunks(archive):
            hash.update(chunk)
    return hash.hexdigest()