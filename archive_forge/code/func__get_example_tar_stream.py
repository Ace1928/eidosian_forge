import struct
import tarfile
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..archive import tar_stream
from ..object_store import MemoryObjectStore
from ..objects import Blob, Tree
from .utils import build_commit_graph
def _get_example_tar_stream(self, *tar_stream_args, **tar_stream_kwargs):
    store = MemoryObjectStore()
    b1 = Blob.from_string(b'somedata')
    store.add_object(b1)
    t1 = Tree()
    t1.add(b'somename', 33188, b1.id)
    store.add_object(t1)
    stream = b''.join(tar_stream(store, t1, *tar_stream_args, **tar_stream_kwargs))
    return BytesIO(stream)