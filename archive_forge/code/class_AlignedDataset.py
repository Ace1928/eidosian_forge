import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import torch
import torch.utils.data
import torchaudio
import tqdm
class AlignedDataset(UnmixDataset):

    def __init__(self, root: str, split: str='train', input_file: str='mixture.wav', output_file: str='vocals.wav', seq_duration: Optional[float]=None, random_chunks: bool=False, sample_rate: float=44100.0, source_augmentations: Optional[Callable]=None, seed: int=42) -> None:
        """A dataset of that assumes multiple track folders
        where each track includes and input and an output file
        which directly corresponds to the the input and the
        output of the model. This dataset is the most basic of
        all datasets provided here, due to the least amount of
        preprocessing, it is also the fastest option, however,
        it lacks any kind of source augmentations or custum mixing.

        Typical use cases:

        * Source Separation (Mixture -> Target)
        * Denoising (Noisy -> Clean)
        * Bandwidth Extension (Low Bandwidth -> High Bandwidth)

        Example
        =======
        data/train/01/mixture.wav --> input
        data/train/01/vocals.wav ---> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_chunks = random_chunks
        self.input_file = input_file
        self.output_file = output_file
        self.tuple_paths = list(self._get_paths())
        if not self.tuple_paths:
            raise RuntimeError('Dataset is empty, please check parameters')
        self.seed = seed
        random.seed(self.seed)

    def __getitem__(self, index):
        input_path, output_path = self.tuple_paths[index]
        if self.random_chunks:
            input_info = load_info(input_path)
            output_info = load_info(output_path)
            duration = min(input_info['duration'], output_info['duration'])
            start = random.uniform(0, duration - self.seq_duration)
        else:
            start = 0
        X_audio, _ = load_audio(input_path, start=start, dur=self.seq_duration)
        Y_audio, _ = load_audio(output_path, start=start, dur=self.seq_duration)
        return (X_audio, Y_audio)

    def __len__(self):
        return len(self.tuple_paths)

    def _get_paths(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                input_path = list(track_path.glob(self.input_file))
                output_path = list(track_path.glob(self.output_file))
                if input_path and output_path:
                    if self.seq_duration is not None:
                        input_info = load_info(input_path[0])
                        output_info = load_info(output_path[0])
                        min_duration = min(input_info['duration'], output_info['duration'])
                        if min_duration > self.seq_duration:
                            yield (input_path[0], output_path[0])
                    else:
                        yield (input_path[0], output_path[0])