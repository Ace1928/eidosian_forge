import time
import numpy as np
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.summary import _output
Adds a scalar summary.

        Args:
          tag: string tag used to uniquely identify this time series.
          data: numeric scalar value for this data point. Accepts any value that
            can be converted to a `np.float32` scalar.
          step: integer step value for this data point. Accepts any value that
            can be converted to a `np.int64` scalar.
          wall_time: optional `float` seconds since the Unix epoch, representing
            the real-world timestamp for this data point. Defaults to None in
            which case the current time will be used.
          description: optional string description for this entire time series.
            This should be constant for a given tag; only the first value
            encountered will be used.
        