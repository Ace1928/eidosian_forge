import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _index(self, plugin_name, run_tag_filter, data_class_filter):
    """List time series and metadata matching the given filters.

        This is like `_list`, but doesn't traverse `Tensors(...)` to
        compute metadata that's not always needed.

        Args:
          plugin_name: A string plugin name filter (required).
          run_tag_filter: An `provider.RunTagFilter`, or `None`.
          data_class_filter: A `summary_pb2.DataClass` filter (required).

        Returns:
          A nested dict `d` such that `d[run][tag]` is a
          `SummaryMetadata` proto.
        """
    if run_tag_filter is None:
        run_tag_filter = provider.RunTagFilter(runs=None, tags=None)
    runs = run_tag_filter.runs
    tags = run_tag_filter.tags
    if runs and len(runs) == 1 and tags and (len(tags) == 1):
        run, = runs
        tag, = tags
        try:
            metadata = self._multiplexer.SummaryMetadata(run, tag)
        except KeyError:
            return {}
        all_metadata = {run: {tag: metadata}}
    else:
        all_metadata = self._multiplexer.AllSummaryMetadata()
    result = {}
    for run, tag_to_metadata in all_metadata.items():
        if runs is not None and run not in runs:
            continue
        result_for_run = {}
        for tag, metadata in tag_to_metadata.items():
            if tags is not None and tag not in tags:
                continue
            if metadata.data_class != data_class_filter:
                continue
            if metadata.plugin_data.plugin_name != plugin_name:
                continue
            result[run] = result_for_run
            result_for_run[tag] = metadata
    return result