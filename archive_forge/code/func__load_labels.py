import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform
def _load_labels(file: Path, subset: str):
    """Load transcirpt, iob, and intent labels for all utterances.

    Args:
        file (Path): The path to the label file.
        subset (str): Subset of the dataset to use. Options: [``"train"``, ``"valid"``, ``"test"``].

    Returns:
        Dictionary of labels, where the key is the filename of the audio,
            and the label is a Tuple of transcript, Inside–outside–beginning (IOB) label, and intention label.
    """
    labels = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            index = line[0]
            trans, iob_intent = ' '.join(line[1:]).split('\t')
            trans = ' '.join(trans.split(' ')[1:-1])
            iob = ' '.join(iob_intent.split(' ')[1:-1])
            intent = iob_intent.split(' ')[-1]
            if subset in index:
                labels[index] = (trans, iob, intent)
    return labels