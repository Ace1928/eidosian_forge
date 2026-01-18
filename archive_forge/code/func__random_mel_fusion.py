import copy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging
def _random_mel_fusion(self, mel, total_frames, chunk_frames):
    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
    if len(ranges[1]) == 0:
        ranges[1] = [0]
    if len(ranges[2]) == 0:
        ranges[2] = [0]
    idx_front = np.random.choice(ranges[0])
    idx_middle = np.random.choice(ranges[1])
    idx_back = np.random.choice(ranges[2])
    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
    mel = torch.tensor(mel[None, None, :])
    mel_shrink = torch.nn.functional.interpolate(mel, size=[chunk_frames, 64], mode='bilinear', align_corners=False)
    mel_shrink = mel_shrink[0][0].numpy()
    mel_fusion = np.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], axis=0)
    return mel_fusion