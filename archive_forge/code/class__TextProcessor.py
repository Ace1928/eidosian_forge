from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torchaudio.models import Tacotron2
class _TextProcessor(ABC):

    @property
    @abstractmethod
    def tokens(self):
        """The tokens that the each value in the processed tensor represent.

        :type: List[str]
        """

    @abstractmethod
    def __call__(self, texts: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
        """Encode the given (batch of) texts into numerical tensors

        Args:
            text (str or list of str): The input texts.

        Returns:
            (Tensor, Tensor):
            Tensor:
                The encoded texts. Shape: `(batch, max length)`
            Tensor:
                The valid length of each sample in the batch. Shape: `(batch, )`.
        """