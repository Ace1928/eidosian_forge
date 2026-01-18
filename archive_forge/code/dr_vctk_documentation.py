from pathlib import Path
from typing import Dict, Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip
Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Clean waveform
            int:
                Sample rate of the clean waveform
            Tensor:
                Noisy waveform
            int:
                Sample rate of the noisy waveform
            str:
                Speaker ID
            str:
                Utterance ID
            str:
                Source
            int:
                Channel ID
        