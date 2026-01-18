import os
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat
class TFRecordTestBase(test_base.DatasetTestBase):
    """Base class for TFRecord-based tests."""

    def setUp(self):
        super(TFRecordTestBase, self).setUp()
        self._num_files = 2
        self._num_records = 7
        self._filenames = self._createFiles()

    def _interleave(self, iterators, cycle_length):
        pending_iterators = iterators
        open_iterators = []
        num_open = 0
        for i in range(cycle_length):
            if pending_iterators:
                open_iterators.append(pending_iterators.pop(0))
                num_open += 1
        while num_open:
            for i in range(min(cycle_length, len(open_iterators))):
                if open_iterators[i] is None:
                    continue
                try:
                    yield next(open_iterators[i])
                except StopIteration:
                    if pending_iterators:
                        open_iterators[i] = pending_iterators.pop(0)
                    else:
                        open_iterators[i] = None
                        num_open -= 1

    def _next_expected_batch(self, file_indices, batch_size, num_epochs, cycle_length, drop_final_batch, use_parser_fn):

        def _next_record(file_indices):
            for j in file_indices:
                for i in range(self._num_records):
                    yield (j, i)

        def _next_record_interleaved(file_indices, cycle_length):
            return self._interleave([_next_record([i]) for i in file_indices], cycle_length)
        record_batch = []
        batch_index = 0
        for _ in range(num_epochs):
            if cycle_length == 1:
                next_records = _next_record(file_indices)
            else:
                next_records = _next_record_interleaved(file_indices, cycle_length)
            for f, r in next_records:
                record = self._record(f, r)
                if use_parser_fn:
                    record = record[1:]
                record_batch.append(record)
                batch_index += 1
                if len(record_batch) == batch_size:
                    yield record_batch
                    record_batch = []
                    batch_index = 0
        if record_batch and (not drop_final_batch):
            yield record_batch

    def _verify_records(self, outputs, batch_size, file_index, num_epochs, interleave_cycle_length, drop_final_batch, use_parser_fn):
        if file_index is not None:
            if isinstance(file_index, list):
                file_indices = file_index
            else:
                file_indices = [file_index]
        else:
            file_indices = range(self._num_files)
        for expected_batch in self._next_expected_batch(file_indices, batch_size, num_epochs, interleave_cycle_length, drop_final_batch, use_parser_fn):
            actual_batch = self.evaluate(outputs())
            self.assertAllEqual(expected_batch, actual_batch)

    def _record(self, f, r):
        return compat.as_bytes('Record %d of file %d' % (r, f))

    def _createFiles(self):
        filenames = []
        for i in range(self._num_files):
            fn = os.path.join(self.get_temp_dir(), 'tf_record.%d.txt' % i)
            filenames.append(fn)
            writer = python_io.TFRecordWriter(fn)
            for j in range(self._num_records):
                writer.write(self._record(i, j))
            writer.close()
        return filenames

    def _writeFile(self, name, data):
        filename = os.path.join(self.get_temp_dir(), name)
        writer = python_io.TFRecordWriter(filename)
        for d in data:
            writer.write(compat.as_bytes(str(d)))
        writer.close()
        return filename