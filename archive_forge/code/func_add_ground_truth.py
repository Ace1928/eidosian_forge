import abc
from typing import Any, Dict, List, Optional
from tensorflow.keras.callbacks import Callback  # type: ignore
import wandb
from wandb.sdk.lib import telemetry
@abc.abstractmethod
def add_ground_truth(self, logs: Optional[Dict[str, float]]=None) -> None:
    """Add ground truth data to `data_table`.

        Use this method to write the logic for adding validation/training data to
        `data_table` initialized using `init_data_table` method.

        Example:
            ```python
            for idx, data in enumerate(dataloader):
                self.data_table.add_data(idx, data)
            ```
        This method is called once `on_train_begin` or equivalent hook.
        """
    raise NotImplementedError(f'{self.__class__.__name__}.add_ground_truth')