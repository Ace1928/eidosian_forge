from typing import Any, Dict, Optional
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.accelerators.cpu import _PSUTIL_AVAILABLE
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
Automatically monitors and logs device stats during training, validation and testing stage.
    ``DeviceStatsMonitor`` is a special callback as it requires a ``logger`` to passed as argument to the ``Trainer``.

    Args:
        cpu_stats: if ``None``, it will log CPU stats only if the accelerator is CPU.
            If ``True``, it will log CPU stats regardless of the accelerator.
            If ``False``, it will not log CPU stats regardless of the accelerator.

    Raises:
        MisconfigurationException:
            If ``Trainer`` has no logger.
        ModuleNotFoundError:
            If ``psutil`` is not installed and CPU stats are monitored.

    Example::

        from lightning.pytorch import Trainer
        from pytorch_lightning.callbacks import DeviceStatsMonitor
        device_stats = DeviceStatsMonitor()
        trainer = Trainer(callbacks=[device_stats])

    