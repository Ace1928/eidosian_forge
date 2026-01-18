import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
@staticmethod
def _add_file(repo, tree_id, filename, contents):
    tree = repo[tree_id]
    blob = objects.Blob()
    blob.data = contents.encode('utf-8')
    repo.object_store.add_object(blob)
    tree.add(filename.encode('utf-8'), stat.S_IFREG | 420, blob.id)
    repo.object_store.add_object(tree)
    return tree.id