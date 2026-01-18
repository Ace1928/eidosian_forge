import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import torch
import torch.utils.data
import torchaudio
import tqdm
class MUSDBDataset(UnmixDataset):

    def __init__(self, target: str='vocals', root: str=None, download: bool=False, is_wav: bool=False, subsets: str='train', split: str='train', seq_duration: Optional[float]=6.0, samples_per_track: int=64, source_augmentations: Optional[Callable]=lambda audio: audio, random_track_mix: bool=False, seed: int=42, *args, **kwargs) -> None:
        """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        target : str
            target name of the source to be separated, defaults to ``vocals``.
        root : str
            root path of MUSDB
        download : boolean
            automatically download 7s preview version of MUSDB
        is_wav : boolean
            specify if the WAV version (instead of the MP4 STEMS) are used
        subsets : list-like [str]
            subset str or list of subset. Defaults to ``train``.
        split : str
            use (stratified) track splits for validation split (``valid``),
            defaults to ``train``.
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        seed : int
            control randomness of dataset iterations
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.

        """
        import musdb
        self.seed = seed
        random.seed(seed)
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.mus = musdb.DB(*args, root=root, is_wav=is_wav, split=split, subsets=subsets, download=download, **kwargs)
        self.sample_rate = 44100.0

    def __getitem__(self, index):
        audio_sources = []
        target_ind = None
        track = self.mus.tracks[index // self.samples_per_track]
        if self.split == 'train' and self.seq_duration:
            for k, source in enumerate(self.mus.setup['sources']):
                if source == self.target:
                    target_ind = k
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)
                track.chunk_duration = self.seq_duration
                track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
                audio = torch.as_tensor(track.sources[source].audio.T, dtype=torch.float32)
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)
            stems = torch.stack(audio_sources, dim=0)
            x = stems.sum(0)
            if target_ind is not None:
                y = stems[target_ind]
            else:
                vocind = list(self.mus.setup['sources'].keys()).index('vocals')
                y = x - stems[vocind]
        else:
            x = torch.as_tensor(track.audio.T, dtype=torch.float32)
            y = torch.as_tensor(track.targets[self.target].audio.T, dtype=torch.float32)
        return (x, y)

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track