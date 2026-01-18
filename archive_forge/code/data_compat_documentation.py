import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tensor_util
Convert `old-style` histogram value to `new-style`.

    The "old-style" format can have outermost bucket limits of -DBL_MAX and
    DBL_MAX, which are problematic for visualization. We replace those here
    with the actual min and max values seen in the input data, but then in
    order to avoid introducing "backwards" buckets (where left edge > right
    edge), we first must drop all empty buckets on the left and right ends.
    