from __future__ import annotations
from typing import Union, Optional
from typing_extensions import Literal, Required, TypedDict
class Hyperparameters(TypedDict, total=False):
    batch_size: Union[Literal['auto'], int]
    'Number of examples in each batch.\n\n    A larger batch size means that model parameters are updated less frequently, but\n    with lower variance.\n    '
    learning_rate_multiplier: Union[Literal['auto'], float]
    'Scaling factor for the learning rate.\n\n    A smaller learning rate may be useful to avoid overfitting.\n    '
    n_epochs: Union[Literal['auto'], int]
    'The number of epochs to train the model for.\n\n    An epoch refers to one full cycle through the training dataset.\n    '