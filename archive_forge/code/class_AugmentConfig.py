import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
@dataclass
class AugmentConfig:
    augment_data_factor: int
    instrument_mixups: List[List[str]]
    velocity_mod_pct: List[float]
    transpose_semitones: List[int]
    time_stretch_pct: List[float]
    seed: int
    cfg: VocabConfig

    def __post_init__(self):
        self.validate()
        if len(self.velocity_mod_pct) == 0:
            self.velocity_mod_pct = [0.0]
        if len(self.transpose_semitones) == 0:
            self.transpose_semitones = [0]
        if len(self.time_stretch_pct) == 0:
            self.time_stretch_pct = [0.0]
        self._instrument_mixups_int = [[self.cfg._bin_str_to_int[i] for i in l if i in self.cfg._bin_str_to_int] for l in self.instrument_mixups]
        self._instrument_mixups_int = [l for l in self._instrument_mixups_int if len(l) > 0]
        self._instrument_pool_assignments = {}
        self._mixup_pools = []
        for pool_i, mixup_list in enumerate(self._instrument_mixups_int):
            pool = set()
            for i in mixup_list:
                pool.add(i)
                self._instrument_pool_assignments[i] = pool_i
            self._mixup_pools.append(pool)

    def validate(self):
        if self.augment_data_factor < 1:
            raise ValueError('augment_data_factor must be at least 1')
        used_instruments = set()
        for mixup_list in self.instrument_mixups:
            for n in mixup_list:
                if n in used_instruments:
                    raise ValueError(f'Duplicate instrument name: {n}')
                used_instruments.add(n)

    @classmethod
    def from_json(cls, path: str, cfg: VocabConfig):
        with open(path, 'r') as f:
            config = json.load(f)
        config['cfg'] = cfg
        if 'seed' not in config:
            config['seed'] = random.randint(0, 2 ** 32 - 1)
        return cls(**config)

    def get_augment_values(self, filename: str) -> Iterator[AugmentValues]:
        yield AugmentValues.default()
        rng = random.Random(self.seed + hash(filename))
        for _ in range(int(self.augment_data_factor - 1)):
            randomized_pools = [list(pool) for pool in self._mixup_pools]
            for pool in randomized_pools:
                rng.shuffle(pool)
            instrument_bin_remap = {}
            for i, pool in enumerate(randomized_pools):
                for j, instrument in enumerate(pool):
                    instrument_bin_remap[instrument] = randomized_pools[i - 1][j]
            yield AugmentValues(instrument_bin_remap=instrument_bin_remap, velocity_mod_factor=1.0 + rng.choice(self.velocity_mod_pct), transpose_semitones=rng.choice(self.transpose_semitones), time_stretch_factor=1.0 + rng.choice(self.time_stretch_pct))