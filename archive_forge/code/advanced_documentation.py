import cProfile
import io
import logging
import pstats
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from typing_extensions import override
from pytorch_lightning.profilers.profiler import Profiler

        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

            line_count_restriction: this can be used to limit the number of functions
                reported for each action. either an integer (to select a count of lines),
                or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)

        Raises:
            ValueError:
                If you attempt to stop recording an action which was never started.
        