import os
import shutil
import tempfile
import uuid
from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
def _readAndCheckMetadataFile(self):
    """Read and check the .metadata debug-events file."""
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
        self.assertTrue(reader.tfdbg_run_id())
        self.assertEqual(reader.tensorflow_version(), versions.__version__)
        self.assertTrue(reader.tfdbg_file_version().startswith('debug.Event'))