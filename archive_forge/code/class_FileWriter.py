import os.path
import time
import warnings
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import plugin_asset
from tensorflow.python.summary.writer.event_file_writer import EventFileWriter
from tensorflow.python.summary.writer.event_file_writer_v2 import EventFileWriterV2
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['summary.FileWriter'])
class FileWriter(SummaryToEventTransformer):
    """Writes `Summary` protocol buffers to event files.

  The `FileWriter` class provides a mechanism to create an event file in a
  given directory and add summaries and events to it. The class updates the
  file contents asynchronously. This allows a training program to call methods
  to add data to the file directly from the training loop, without slowing down
  training.

  When constructed with a `tf.compat.v1.Session` parameter, a `FileWriter`
  instead forms a compatibility layer over new graph-based summaries
  to facilitate the use of new summary writing with
  pre-existing code that expects a `FileWriter` instance.

  This class is not thread-safe.

  @compatibility(TF2)
  This API is not compatible with eager execution or `tf.function`. To migrate
  to TF2, please use `tf.summary.create_file_writer` instead for summary
  management. To specify the summary step, you can manage the context with
  `tf.summary.SummaryWriter`, which is returned by
  `tf.summary.create_file_writer()`. Or, you can also use the `step` argument
  of summary functions such as `tf.summary.histogram`.
  See the usage example shown below.

  For a comprehensive `tf.summary` migration guide, please follow
  [Migrating tf.summary usage to
  TF 2.0](https://www.tensorflow.org/tensorboard/migrate#in_tf_1x).

  #### How to Map Arguments

  | TF1 Arg Name        | TF2 Arg Name    | Note                              |
  | :---------------- | :---------------- | :-------------------------------- |
  | `logdir`          | `logdir`          | -                                 |
  | `graph`           | Not supported     | -                                 |
  | `max_queue`       | `max_queue`       | -                                 |
  | `flush_secs`      | `flush_millis`    | The unit of time is changed       |
  :                     :                 : from seconds to milliseconds.     :
  | `graph_def`       | Not supported     | -                                 |
  | `filename_suffix` | `filename_suffix` | -                                 |
  | `name`            | `name`            | -                                 |

  #### TF1 & TF2 Usage Example

  TF1:

  ```python
  dist = tf.compat.v1.placeholder(tf.float32, [100])
  tf.compat.v1.summary.histogram(name="distribution", values=dist)
  writer = tf.compat.v1.summary.FileWriter("/tmp/tf1_summary_example")
  summaries = tf.compat.v1.summary.merge_all()

  sess = tf.compat.v1.Session()
  for step in range(100):
    mean_moving_normal = np.random.normal(loc=step, scale=1, size=[100])
    summ = sess.run(summaries, feed_dict={dist: mean_moving_normal})
    writer.add_summary(summ, global_step=step)
  ```

  TF2:

  ```python
  writer = tf.summary.create_file_writer("/tmp/tf2_summary_example")
  for step in range(100):
    mean_moving_normal = np.random.normal(loc=step, scale=1, size=[100])
    with writer.as_default(step=step):
      tf.summary.histogram(name='distribution', data=mean_moving_normal)
  ```

  @end_compatibility
  """

    def __init__(self, logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None, filename_suffix=None, session=None):
        """Creates a `FileWriter`, optionally shared within the given session.

    Typically, constructing a file writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers constructed when you
    call one of the following functions: `add_summary()`, `add_session_log()`,
    `add_event()`, or `add_graph()`.

    If you pass a `Graph` to the constructor it is added to
    the event file. (This is equivalent to calling `add_graph()` later).

    TensorBoard will pick the graph from the file and display it graphically so
    you can interactively explore the graph you built. You will usually pass
    the graph from the session in which you launched it:

    ```python
    ...create a graph...
    # Launch the graph in a session.
    sess = tf.compat.v1.Session()
    # Create a summary writer, add the 'graph' to the event file.
    writer = tf.compat.v1.summary.FileWriter(<some-directory>, sess.graph)
    ```

    The `session` argument to the constructor makes the returned `FileWriter` a
    compatibility layer over new graph-based summaries (`tf.summary`).
    Crucially, this means the underlying writer resource and events file will
    be shared with any other `FileWriter` using the same `session` and `logdir`.
    In either case, ops will be added to `session.graph` to control the
    underlying file writer resource.

    Args:
      logdir: A string. Directory where event file will be written.
      graph: A `Graph` object, such as `sess.graph`.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      graph_def: DEPRECATED: Use the `graph` argument instead.
      filename_suffix: A string. Every event file's name is suffixed with
        `suffix`.
      session: A `tf.compat.v1.Session` object. See details above.

    Raises:
      RuntimeError: If called with eager execution enabled.

    @compatibility(eager)
      `v1.summary.FileWriter` is not compatible with eager execution.
      To write TensorBoard summaries under eager execution,
      use `tf.summary.create_file_writer` or
      a `with v1.Graph().as_default():` context.
    @end_compatibility
    """
        if context.executing_eagerly():
            raise RuntimeError('v1.summary.FileWriter is not compatible with eager execution. Use `tf.summary.create_file_writer`,or a `with v1.Graph().as_default():` context')
        if session is not None:
            event_writer = EventFileWriterV2(session, logdir, max_queue, flush_secs, filename_suffix)
        else:
            event_writer = EventFileWriter(logdir, max_queue, flush_secs, filename_suffix)
        self._closed = False
        super(FileWriter, self).__init__(event_writer, graph, graph_def)

    def __enter__(self):
        """Make usable with "with" statement."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Make usable with "with" statement."""
        self.close()

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.event_writer.get_logdir()

    def _warn_if_event_writer_is_closed(self):
        if self._closed:
            warnings.warn('Attempting to use a closed FileWriter. The operation will be a noop unless the FileWriter is explicitly reopened.')

    def _add_event(self, event, step):
        self._warn_if_event_writer_is_closed()
        super(FileWriter, self)._add_event(event, step)

    def add_event(self, event):
        """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
        self._warn_if_event_writer_is_closed()
        self.event_writer.add_event(event)

    def flush(self):
        """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
        self._warn_if_event_writer_is_closed()
        self.event_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
        self.event_writer.close()
        self._closed = True

    def reopen(self):
        """Reopens the EventFileWriter.

    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.

    Does nothing if the EventFileWriter was not closed.
    """
        self.event_writer.reopen()
        self._closed = False