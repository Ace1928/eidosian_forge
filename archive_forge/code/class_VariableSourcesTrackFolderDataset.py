import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import torch
import torch.utils.data
import torchaudio
import tqdm
class VariableSourcesTrackFolderDataset(UnmixDataset):

    def __init__(self, root: str, split: str='train', target_file: str='vocals.wav', ext: str='.wav', seq_duration: Optional[float]=None, random_chunks: bool=False, random_interferer_mix: bool=False, sample_rate: float=44100.0, source_augmentations: Optional[Callable]=lambda audio: audio, silence_missing_targets: bool=False) -> None:
        """A dataset that assumes audio sources to be stored
        in track folder where each track has a _variable_ number of sources.
        The users specifies the target file-name (`target_file`)
        and the extension of sources to used for mixing.
        A linear mix is performed on the fly by summing all sources in a
        track folder.

        Since the number of sources differ per track,
        while target is fixed, a random track mix
        augmentation cannot be used. Instead, a random track
        can be used to load the interfering sources.

        Also make sure, that you do not provide the mixture
        file among the sources!

        Example
        =======
        train/1/vocals.wav --> input target           train/1/drums.wav --> input target     |
        train/1/bass.wav --> input target    --+--> input
        train/1/accordion.wav --> input target |
        train/1/marimba.wav --> input target  /

        train/1/vocals.wav -----------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_chunks = random_chunks
        self.random_interferer_mix = random_interferer_mix
        self.source_augmentations = source_augmentations
        self.target_file = target_file
        self.ext = ext
        self.silence_missing_targets = silence_missing_targets
        self.tracks = list(self.get_tracks())

    def __getitem__(self, index):
        target_track_path = self.tracks[index]['path']
        if self.random_chunks:
            target_min_duration = self.tracks[index]['min_duration']
            target_start = random.uniform(0, target_min_duration - self.seq_duration)
        else:
            target_start = 0
        if self.random_interferer_mix:
            random_idx = random.choice(range(len(self.tracks)))
            intfr_track_path = self.tracks[random_idx]['path']
            if self.random_chunks:
                intfr_min_duration = self.tracks[random_idx]['min_duration']
                intfr_start = random.uniform(0, intfr_min_duration - self.seq_duration)
            else:
                intfr_start = 0
        else:
            intfr_track_path = target_track_path
            intfr_start = target_start
        sources = sorted(list(intfr_track_path.glob('*' + self.ext)))
        x = 0
        for source_path in sources:
            if source_path == intfr_track_path / self.target_file:
                continue
            try:
                audio, _ = load_audio(source_path, start=intfr_start, dur=self.seq_duration)
            except RuntimeError:
                index = index - 1 if index > 0 else index + 1
                return self.__getitem__(index)
            x += self.source_augmentations(audio)
        if Path(target_track_path / self.target_file).exists():
            y, _ = load_audio(target_track_path / self.target_file, start=target_start, dur=self.seq_duration)
            y = self.source_augmentations(y)
            x += y
        else:
            y = torch.zeros(audio.shape)
        return (x, y)

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self):
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                if Path(track_path, self.target_file).exists() or self.silence_missing_targets:
                    sources = sorted(list(track_path.glob('*' + self.ext)))
                    if not sources:
                        print('empty track: ', track_path)
                        continue
                    if self.seq_duration is not None:
                        infos = list(map(load_info, sources))
                        min_duration = min((i['duration'] for i in infos))
                        if min_duration > self.seq_duration:
                            yield {'path': track_path, 'min_duration': min_duration}
                    else:
                        yield {'path': track_path, 'min_duration': None}