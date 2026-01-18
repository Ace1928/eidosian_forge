import csv
import os
from pathlib import Path
from typing import Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform
Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                File name
            int:
                Speaker ID
            str:
                Transcription
            str:
                Action
            str:
                Object
            str:
                Location
        