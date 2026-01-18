import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import torch
import torch.utils.data
import torchaudio
import tqdm
class FixedSourcesTrackFolderDataset(UnmixDataset):

    def __init__(self, root: str, split: str='train', target_file: str='vocals.wav', interferer_files: List[str]=['bass.wav', 'drums.wav'], seq_duration: Optional[float]=None, random_chunks: bool=False, random_track_mix: bool=False, source_augmentations: Optional[Callable]=lambda audio: audio, sample_rate: float=44100.0, seed: int=42) -> None:
        """A dataset that assumes audio sources to be stored
        in track folder where each track has a fixed number of sources.
        For each track the users specifies the target file-name (`target_file`)
        and a list of interferences files (`interferer_files`).
        A linear mix is performed on the fly by summing the target and
        the inferers up.

        Due to the fact that all tracks comprise the exact same set
        of sources, the random track mixing augmentation technique
        can be used, where sources from different tracks are mixed
        together. Setting `random_track_mix=True` results in an
        unaligned dataset.
        When random track mixing is enabled, we define an epoch as
        when the the target source from all tracks has been seen and only once
        with whatever interfering sources has randomly been drawn.

        This dataset is recommended to be used for small/medium size
        for example like the MUSDB18 or other custom source separation
        datasets.

        Example
        =======
        train/1/vocals.wav ---------------        train/1/drums.wav (interferer1) ---+--> input
        train/1/bass.wav -(interferer2) --/

        train/1/vocals.wav -------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_track_mix = random_track_mix
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        self.target_file = target_file
        self.interferer_files = interferer_files
        self.source_files = self.interferer_files + [self.target_file]
        self.seed = seed
        random.seed(self.seed)
        self.tracks = list(self.get_tracks())
        if not len(self.tracks):
            raise RuntimeError('No tracks found')

    def __getitem__(self, index):
        track_path = self.tracks[index]['path']
        min_duration = self.tracks[index]['min_duration']
        if self.random_chunks:
            start = random.uniform(0, min_duration - self.seq_duration)
        else:
            start = 0
        audio_sources = []
        target_audio, _ = load_audio(track_path / self.target_file, start=start, dur=self.seq_duration)
        target_audio = self.source_augmentations(target_audio)
        audio_sources.append(target_audio)
        for source in self.interferer_files:
            if self.random_track_mix:
                random_idx = random.choice(range(len(self.tracks)))
                track_path = self.tracks[random_idx]['path']
                if self.random_chunks:
                    min_duration = self.tracks[random_idx]['min_duration']
                    start = random.uniform(0, min_duration - self.seq_duration)
            audio, _ = load_audio(track_path / source, start=start, dur=self.seq_duration)
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)
        stems = torch.stack(audio_sources)
        x = stems.sum(0)
        y = stems[0]
        return (x, y)

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                source_paths = [track_path / s for s in self.source_files]
                if not all((sp.exists() for sp in source_paths)):
                    print('Exclude track ', track_path)
                    continue
                if self.seq_duration is not None:
                    infos = list(map(load_info, source_paths))
                    min_duration = min((i['duration'] for i in infos))
                    if min_duration > self.seq_duration:
                        yield {'path': track_path, 'min_duration': min_duration}
                else:
                    yield {'path': track_path, 'min_duration': None}