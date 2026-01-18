from typing import Any, Iterator, List, Optional
from typing_extensions import override
from lightning_fabric.utilities.data import sized_len
from pytorch_lightning.utilities.combined_loader import _ITERATOR_RETURN, CombinedLoader
from pytorch_lightning.utilities.exceptions import MisconfigurationException
This class is used to return directly the `dataloader_iter` to the ``LightningModule`` training_step for users
    to implement their own pre-fetching logic. This feature can be activated as follows:

    Example::

        Class MyModel(LightningModule):
            def training_step(self, dataloader_iter: Iterator) -> None:
                # it is the user responsibility to fetch and move the batch to the right device.
                batch, batch_idx, dataloader_idx = next(dataloader_iter)
                batch = batch.to(self.device)
                ...

    