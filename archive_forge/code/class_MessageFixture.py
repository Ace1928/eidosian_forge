from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
class MessageFixture(IpcFixture):

    def _get_writer(self, sink, schema):
        return pa.RecordBatchStreamWriter(sink, schema)