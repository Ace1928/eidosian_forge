import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
def _commit_file_with_content(repo, filename, content):
    file_path = os.path.join(repo.path, filename)
    with open(file_path, 'w') as f:
        f.write(content)
    porcelain.add(repo, paths=[file_path])
    sha = porcelain.commit(repo, message=b'add ' + filename.encode(), committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
    return (sha, file_path)