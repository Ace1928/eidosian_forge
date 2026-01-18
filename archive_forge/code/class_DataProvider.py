from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
class DataProvider(metaclass=abc.ABCMeta):
    """Interface for reading TensorBoard scalar, tensor, and blob data.

    These APIs are under development and subject to change. For instance,
    providers may be asked to implement more filtering mechanisms, such as
    downsampling strategies or domain restriction by step or wall time.

    The data provider interface specifies three *data classes*: scalars,
    tensors, and blob sequences. All data is stored in *time series* for
    one of these data classes. A time series is identified by run name and
    tag name (each a non-empty text string), as well as an experiment ID
    and plugin name (see below). Points in a time series are uniquely
    indexed by *step*, an arbitrary non-negative integer. Each point in a
    time series also has an associated wall time, plus its actual value,
    which is drawn from the corresponding data class.

    Each point in a scalar time series contains a single scalar value, as
    a 64-bit floating point number. Scalars are "privileged" rather than
    being subsumed under tensors because there are useful operations on
    scalars that don't make sense in the general tensor case: e.g., "list
    all scalar time series with tag name `accuracy` whose exponentially
    weighted moving average is at least 0.999".

    Each point in a tensor time series contains a tensor of arbitrary
    dtype (including byte strings and text strings) and shape (including
    rank-0 tensors, a.k.a. scalars). Each tensor is expected to be
    "reasonably small" to accommodate common database cell size limits.
    For instance, a histogram with a bounded number of buckets (say, 30)
    occupies about 500 bytes, and a PR curve with a bounded number of
    thresholds (say, 201) occupies about 5000 bytes. These are both well
    within typical database tolerances (Google Cloud Spanner: 10 MiB;
    MySQL: 64 KiB), and would be appropriate to store as tensors. By
    contrast, image, audio, or model graph data may easily be multiple
    megabytes in size, and so should be stored as blobs instead. The
    tensors at each step in a time series need not have the same dtype or
    shape.

    Each point in a blob sequence time series contains an ordered sequence
    of zero or more blobs, which are arbitrary data with no tensor
    structure. These might represent PNG-encoded image data, protobuf wire
    encodings of TensorFlow graphs, or PLY-format 3D mesh data, for some
    examples. This data class provides blob *sequences* rather than just
    blobs because it's common to want to take multiple homogeneous samples
    of a given time series: say, "show me the bounding box classifications
    for 3 random inputs from this batch". A single blob can of course be
    represented as a blob sequence that always has exactly one element.

    When reading time series, *downsampling* refers to selecting a
    subset of the points in each time series. Downsampling only occurs
    across the step axis (rather than, e.g., the blobs in a single blob
    sequence datum), and occurs individually within each time series.
    When downsampling, the latest datum should always be included in the
    sample, so that clients have a view of metrics that is maximally up
    to date. Implementations may choose to force the first (oldest)
    datum to be included in each sample as well, but this is not
    required; clients should not make assumptions either way. The
    remainder of the points in the sample should be selected uniformly
    at random from available points. Downsampling should be
    deterministic within a time series. It is also useful for the
    downsampling behavior to depend only on the set of step values
    within a time series, such that two "parallel" time series with data
    at exactly the same steps also retain the same steps after
    downsampling.

    Every time series belongs to a specific experiment and is owned by a
    specific plugin. (Thus, the "primary key" for a time series has four
    components: experiment, plugin, run, tag.) The experiment ID is an
    arbitrary URL-safe non-empty text string, whose interpretation is at
    the discretion of the data provider. As a special case, the empty
    string as an experiment ID denotes that no experiment was given. Data
    providers may or may not fully support an empty experiment ID. The
    plugin name should correspond to the `plugin_data.plugin_name` field
    of the `SummaryMetadata` proto passed to `tf.summary.write`.

    Additionally, the data provider interface specifies one *hyperparameter*
    class, which is metadata about the parameters used to generate the data for
    one or more runs within one or more experiments. Each hyperparameter has a
    value type -- one of string, bool, and float. Each one also has a domain,
    which describes the set of known values for that hyperparameter across the
    given set of experiments.

    There is a corresponding *hyperparameter value* class, which describes an
    actual value of a hyperparameter that was logged during experiment
    execution.

    Each run within an experiment may specify its own value for a
    hyperparameter. Runs that were logically executed together with the same set
    of hyperparameter values form a hyperparameter session group. When querying
    for hyperparameter values for a set of experiments, the result will group
    runs by hyperparameter session group and provide one set of hyperparameter
    values for each group.

    All methods on this class take a `RequestContext` parameter as the
    first positional argument. This argument is temporarily optional to
    facilitate migration, but will be required in the future.

    Unless otherwise noted, any methods on this class may raise errors
    defined in `tensorboard.errors`, like `tensorboard.errors.NotFoundError`.

    If not implemented, optional methods may return `None`.
    """

    def experiment_metadata(self, ctx=None, *, experiment_id):
        """Retrieve metadata of a given experiment.

        The metadata may include fields such as name and description
        of the experiment, as well as a timestamp for the experiment.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id:  ID of the experiment in question.

        Returns:
          An `ExperimentMetadata` object containing metadata about the
          experiment.
        """
        return ExperimentMetadata()

    def list_plugins(self, ctx=None, *, experiment_id):
        """List all plugins that own data in a given experiment.

        This should be the set of all plugin names `p` such that calling
        `list_scalars`, `list_tensors`, or `list_blob_sequences` for the
        given `experiment_id` and plugin name `p` gives a non-empty
        result.

        This operation is optional, but may later become required.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.

        Returns:
          A collection of strings representing plugin names, or `None`
          if this operation is not supported by this data provider.
        """
        return None

    @abc.abstractmethod
    def list_runs(self, ctx=None, *, experiment_id):
        """List all runs within an experiment.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.

        Returns:
          A collection of `Run` values.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    @abc.abstractmethod
    def list_scalars(self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None):
        """List metadata about scalar time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created
            the data to be queried. Required.
          run_tag_filter: Optional `RunTagFilter` value. If omitted, all
            runs and tags will be included.

        The result will only contain keys for run-tag combinations that
        actually exist, which may not include all entries in the
        `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a `ScalarTimeSeries`
          value.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    @abc.abstractmethod
    def read_scalars(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
        """Read values from scalar time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created
            the data to be queried. Required.
          downsample: Integer number of steps to which to downsample the
            results (e.g., `1000`). See `DataProvider` class docstring
            for details about this parameter. Required.
          run_tag_filter: Optional `RunTagFilter` value. If provided, a time
            series will only be included in the result if its run and tag
            both pass this filter. If `None`, all time series will be
            included.

        The result will only contain keys for run-tag combinations that
        actually exist, which may not include all entries in the
        `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a list of
          `ScalarDatum` values sorted by step.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def list_tensors(self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None):
        """List metadata about tensor time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created
            the data to be queried. Required.
          run_tag_filter: Optional `RunTagFilter` value. If omitted, all
            runs and tags will be included.

        The result will only contain keys for run-tag combinations that
        actually exist, which may not include all entries in the
        `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a `TensorTimeSeries`
          value.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def read_tensors(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
        """Read values from tensor time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created
            the data to be queried. Required.
          downsample: Integer number of steps to which to downsample the
            results (e.g., `1000`). See `DataProvider` class docstring
            for details about this parameter. Required.
          run_tag_filter: Optional `RunTagFilter` value. If provided, a time
            series will only be included in the result if its run and tag
            both pass this filter. If `None`, all time series will be
            included.

        The result will only contain keys for run-tag combinations that
        actually exist, which may not include all entries in the
        `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a list of
          `TensorDatum` values sorted by step.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def list_blob_sequences(self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None):
        """List metadata about blob sequence time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created the data
            to be queried. Required.
          run_tag_filter: Optional `RunTagFilter` value. If omitted, all runs and
            tags will be included. The result will only contain keys for run-tag
            combinations that actually exist, which may not include all entries in
            the `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a `BlobSequenceTimeSeries`
          value.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def read_blob_sequences(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
        """Read values from blob sequence time series.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_id: ID of enclosing experiment.
          plugin_name: String name of the TensorBoard plugin that created the data
            to be queried. Required.
          downsample: Integer number of steps to which to downsample the
            results (e.g., `1000`). See `DataProvider` class docstring
            for details about this parameter. Required.
          run_tag_filter: Optional `RunTagFilter` value. If provided, a time series
            will only be included in the result if its run and tag both pass this
            filter. If `None`, all time series will be included. The result will
            only contain keys for run-tag combinations that actually exist, which
            may not include all entries in the `run_tag_filter`.

        Returns:
          A nested map `d` such that `d[run][tag]` is a list of
          `BlobSequenceDatum` values sorted by step.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def read_blob(self, ctx=None, *, blob_key):
        """Read data for a single blob.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          blob_key: A key identifying the desired blob, as provided by
            `read_blob_sequences(...)`.

        Returns:
          Raw binary data as `bytes`.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        pass

    def list_hyperparameters(self, ctx=None, *, experiment_ids):
        """List hyperparameters metadata.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_ids: A Collection[string] of IDs of the enclosing
            experiments.

        Returns:
          A Collection[Hyperparameter] describing the hyperparameter metadata
          for the experiments.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        return []

    def read_hyperparameters(self, ctx=None, *, experiment_ids, filters=None, sort=None):
        """Read hyperparameter values.

        Args:
          ctx: A TensorBoard `RequestContext` value.
          experiment_ids: A Collection[string] of IDs of the enclosing
            experiments.
          filters: A Collection[HyperparameterFilter] that constrain the
            returned session groups based on hyperparameter value.
          sort: A Sequence[HyperparameterSort] that specify how the results
            should be sorted.

        Returns:
          A Collection[HyperparameterSessionGroup] describing the groups and
          their hyperparameter values.

        Raises:
          tensorboard.errors.PublicError: See `DataProvider` class docstring.
        """
        return []