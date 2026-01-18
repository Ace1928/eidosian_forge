import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
class DebugEventsReader:
    """Reader class for a tfdbg v2 DebugEvents directory."""
    _READER_RELEASE_PER = 100
    _METADATA_SUFFIX = '.metadata'
    _SOURCE_FILE_SUFFIX = '.source_files'
    _STACK_FRAMES_SUFFIX = '.stack_frames'
    _GRAPHS_SUFFIX = '.graphs'
    _EXECUTION_SUFFIX = '.execution'
    _GRAPH_EXECUTION_TRACES_SUFFIX = '.graph_execution_traces'

    def __init__(self, dump_root):
        if not file_io.is_directory(dump_root):
            raise ValueError('Specified dump_root is not a directory: %s' % dump_root)
        self._dump_root = dump_root
        self._metadata_paths = self._load_metadata_files()
        prefixes = [metadata_path[:-len(self._METADATA_SUFFIX)] for metadata_path in self._metadata_paths]
        prefix = prefixes[0]
        self._source_files_path = compat.as_bytes(prefix + self._SOURCE_FILE_SUFFIX)
        self._stack_frames_path = compat.as_bytes(prefix + self._STACK_FRAMES_SUFFIX)
        self._graphs_path = compat.as_bytes(prefix + self._GRAPHS_SUFFIX)
        self._execution_path = compat.as_bytes(prefix + self._EXECUTION_SUFFIX)
        self._graph_execution_traces_paths = [compat.as_bytes(prefix + self._GRAPH_EXECUTION_TRACES_SUFFIX) for prefix in prefixes]
        self._readers = dict()
        self._reader_offsets = dict()
        self._readers_lock = threading.Lock()
        self._reader_read_locks = dict()
        self._offsets = dict()

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

    def starting_wall_time(self):
        """Get the starting timestamp of the instrumented TensorFlow program.

    When there are multiple hosts (i.e., multiple tfdbg file sets), the earliest
    timestamp among the file sets is returned. It is assumed to be the job that
    starts first (e.g., the coordinator).

    Returns:
      Starting timestamp in seconds since the epoch, as a float.
    """
        return self._starting_wall_time

    def tfdbg_run_id(self):
        """Get the run ID of the instrumented TensorFlow program."""
        return self._tfdbg_run_id

    def tensorflow_version(self):
        """Get the version string of TensorFlow that the debugged program ran on."""
        return self._tensorflow_version

    def tfdbg_file_version(self):
        """Get the tfdbg file format version."""
        return self._file_version

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        del exception_type, exception_value, traceback
        self.close()

    def _generic_iterator(self, file_path):
        """A helper method that makes an iterator given a debug-events file path.

    Repeated calls to this method create iterators that remember the last
    successful reading position (offset) for each given `file_path`. So the
    iterators are meant for incremental reading of the file.

    Args:
      file_path: Path to the file to create the iterator for.

    Yields:
      A tuple of (offset, debug_event_proto) on each `next()` call.
    """
        yield_count = 0
        reader = self._get_reader(file_path)
        read_lock = self._reader_read_locks[file_path]
        read_lock.acquire()
        try:
            while True:
                current_offset = self._reader_offsets[file_path]
                try:
                    record, self._reader_offsets[file_path] = reader.read(current_offset)
                except (errors.DataLossError, IndexError):
                    break
                yield DebugEventWithOffset(debug_event=debug_event_pb2.DebugEvent.FromString(record), offset=current_offset)
                yield_count += 1
                if yield_count % self._READER_RELEASE_PER == 0:
                    read_lock.release()
                    read_lock.acquire()
        finally:
            read_lock.release()

    def _get_reader(self, file_path):
        """Get a random-access reader for TFRecords file at file_path."""
        file_path = compat.as_bytes(file_path)
        if file_path not in self._readers:
            with self._readers_lock:
                if file_path not in self._readers:
                    self._readers[file_path] = tf_record.tf_record_random_reader(file_path)
                    self._reader_read_locks[file_path] = threading.Lock()
                    self._reader_offsets[file_path] = 0
        return self._readers[file_path]

    def source_files_iterator(self):
        return self._generic_iterator(self._source_files_path)

    def stack_frames_iterator(self):
        return self._generic_iterator(self._stack_frames_path)

    def graphs_iterator(self):
        return self._generic_iterator(self._graphs_path)

    def read_source_files_event(self, offset):
        """Read a DebugEvent proto at given offset from the .source_files file."""
        with self._reader_read_locks[self._source_files_path]:
            proto_string = self._get_reader(self._source_files_path).read(offset)[0]
        return debug_event_pb2.DebugEvent.FromString(proto_string)

    def read_graphs_event(self, offset):
        """Read a DebugEvent proto at a given offset from the .graphs file.

    Args:
      offset: Offset to read the DebugEvent proto from.

    Returns:
      A DebugEventProto.

    Raises:
      `errors.DataLossError` if offset is at a wrong location.
      `IndexError` if offset is out of range of the file.
    """
        return debug_event_pb2.DebugEvent.FromString(self._get_reader(self._graphs_path).read(offset)[0])

    def execution_iterator(self):
        return self._generic_iterator(self._execution_path)

    def read_execution_event(self, offset):
        """Read a DebugEvent proto at a given offset from the .execution file.

    Args:
      offset: Offset to read the DebugEvent proto from.

    Returns:
      A DebugEventProto.

    Raises:
      `errors.DataLossError` if offset is at a wrong location.
      `IndexError` if offset is out of range of the file.
    """
        with self._reader_read_locks[self._execution_path]:
            proto_string = self._get_reader(self._execution_path).read(offset)[0]
        return debug_event_pb2.DebugEvent.FromString(proto_string)

    def graph_execution_traces_iterators(self):
        return [self._generic_iterator(path) for path in self._graph_execution_traces_paths]

    def read_graph_execution_traces_event(self, locator):
        """Read DebugEvent at given offset from given .graph_execution_traces file.

    Args:
      locator: A (file_index, offset) tuple that locates the DebugEvent
        containing the graph execution trace.

    Returns:
      A DebugEventProto.

    Raises:
      `errors.DataLossError` if offset is at a wrong location.
      `IndexError` if offset is out of range of the file.
    """
        file_index, offset = locator
        graph_execution_traces_path = self._graph_execution_traces_paths[file_index]
        with self._reader_read_locks[graph_execution_traces_path]:
            proto_string = self._get_reader(graph_execution_traces_path).read(offset)[0]
        return debug_event_pb2.DebugEvent.FromString(proto_string)

    def close(self):
        with self._readers_lock:
            file_paths = list(self._readers.keys())
            for file_path in file_paths:
                self._readers[file_path].close()
                del self._readers[file_path]