import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tensor_util
def _migrate_histogram_value(value):
    """Convert `old-style` histogram value to `new-style`.

    The "old-style" format can have outermost bucket limits of -DBL_MAX and
    DBL_MAX, which are problematic for visualization. We replace those here
    with the actual min and max values seen in the input data, but then in
    order to avoid introducing "backwards" buckets (where left edge > right
    edge), we first must drop all empty buckets on the left and right ends.
    """
    histogram_value = value.histo
    bucket_counts = histogram_value.bucket
    n = len(bucket_counts)
    start = next((i for i in range(n) if bucket_counts[i] > 0), n)
    end = next((i for i in reversed(range(n)) if bucket_counts[i] > 0), -1)
    if start > end:
        buckets = np.zeros([0, 3], dtype=np.float32)
    else:
        bucket_counts = bucket_counts[start:end + 1]
        inner_edges = histogram_value.bucket_limit[start:end]
        bucket_lefts = [histogram_value.min] + inner_edges
        bucket_rights = inner_edges + [histogram_value.max]
        buckets = np.array([bucket_lefts, bucket_rights, bucket_counts], dtype=np.float32).transpose()
    summary_metadata = histogram_metadata.create_summary_metadata(display_name=value.metadata.display_name or value.tag, description=value.metadata.summary_description)
    return make_summary(value.tag, summary_metadata, buckets)