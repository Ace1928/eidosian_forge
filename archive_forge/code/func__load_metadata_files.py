import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _load_metadata_files(self):
    """Load and parse metadata files in the dump root.

    Check that all metadata files have a common tfdbg_run_id, and raise
    a ValueError if their tfdbg_run_ids differ.

    Returns:
      A list of metadata file paths in ascending order of their starting
        wall_time timestamp.
    """
    metadata_paths = file_io.get_matching_files(os.path.join(self._dump_root, '*%s' % self._METADATA_SUFFIX))
    if not metadata_paths:
        raise ValueError('Cannot find any tfdbg metadata file in directory: %s' % self._dump_root)
    wall_times = []
    run_ids = []
    tensorflow_versions = []
    file_versions = []
    for metadata_path in metadata_paths:
        reader = tf_record.tf_record_random_reader(metadata_path)
        try:
            record = reader.read(0)[0]
            debug_event = debug_event_pb2.DebugEvent.FromString(record)
            wall_times.append(debug_event.wall_time)
            run_ids.append(debug_event.debug_metadata.tfdbg_run_id)
            tensorflow_versions.append(debug_event.debug_metadata.tensorflow_version)
            file_versions.append(debug_event.debug_metadata.file_version)
        finally:
            reader.close()
    self._starting_wall_time = wall_times[0]
    self._tfdbg_run_id = run_ids[0]
    self._tensorflow_version = tensorflow_versions[0]
    self._file_version = file_versions[0]
    if len(metadata_paths) == 1:
        return metadata_paths
    num_no_id = len([run_id for run_id in run_ids if not run_id])
    if num_no_id:
        paths_without_run_id = [metadata_path for metadata_path, run_id in zip(metadata_paths, run_ids) if not run_id]
        raise ValueError('Found %d tfdbg metadata files and %d of them do not have tfdbg run ids. The metadata files without run ids are: %s' % (len(run_ids), num_no_id, paths_without_run_id))
    elif len(set(run_ids)) != 1:
        raise ValueError('Unexpected: Found multiple (%d) tfdbg2 runs in directory %s' % (len(set(run_ids)), self._dump_root))
    paths_and_timestamps = sorted(zip(metadata_paths, wall_times), key=lambda t: t[1])
    self._starting_wall_time = paths_and_timestamps[0][1]
    return [path[0] for path in paths_and_timestamps]