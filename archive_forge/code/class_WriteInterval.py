from typing import Any, Literal, Optional, Sequence
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
class WriteInterval(LightningEnum):
    BATCH = 'batch'
    EPOCH = 'epoch'
    BATCH_AND_EPOCH = 'batch_and_epoch'

    @property
    def on_batch(self) -> bool:
        return self in (self.BATCH, self.BATCH_AND_EPOCH)

    @property
    def on_epoch(self) -> bool:
        return self in (self.EPOCH, self.BATCH_AND_EPOCH)