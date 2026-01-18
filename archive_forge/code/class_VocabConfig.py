import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
@dataclass
class VocabConfig:
    note_events: int
    wait_events: int
    max_wait_time: int
    velocity_events: int
    velocity_bins: int
    velocity_exp: float
    do_token_sorting: bool
    unrolled_tokens: bool
    decode_end_held_note_delay: float
    decode_fix_repeated_notes: bool
    bin_instrument_names: List[str]
    ch10_instrument_bin_name: str
    program_name_to_bin_name: Dict[str, str]
    bin_name_to_program_name: Dict[str, str]
    instrument_names: Dict[str, str]
    velocity_bins_override: Optional[List[int]] = None

    def __post_init__(self):
        self.validate()
        self._instrument_names_str_to_int = {name: int(i) for i, name in self.instrument_names.items()}
        self._instrument_names_int_to_str = {int(i): name for i, name in self.instrument_names.items()}
        self._bin_str_to_int = {name: int(i) for i, name in enumerate(self.bin_instrument_names)}
        self._bin_int_to_instrument_int = [self._instrument_names_str_to_int[self.bin_name_to_program_name[name]] if name != self.ch10_instrument_bin_name else 0 for name in self.bin_instrument_names]
        self._instrument_int_to_bin_int = [self._bin_str_to_int[self.program_name_to_bin_name[instr]] if self.program_name_to_bin_name[instr] != '' else -1 for instr in self.program_name_to_bin_name.keys()]
        self._ch10_bin_int = self._bin_str_to_int[self.ch10_instrument_bin_name] if self.ch10_instrument_bin_name else -1
        self.short_instr_bin_names = []
        for instr in self.bin_instrument_names:
            i = min(1, len(instr))
            while instr[:i] in self.short_instr_bin_names:
                i += 1
            self.short_instr_bin_names.append(instr[:i])
        self._short_instrument_names_str_to_int = {name: int(i) for i, name in enumerate(self.short_instr_bin_names)}
        range_excluding_ch10 = [i if i < 9 else i + 1 for i in range(len(self.bin_instrument_names))]
        bins_excluding_ch10 = [n for n in self.bin_instrument_names if n != self.ch10_instrument_bin_name]
        self.bin_channel_map = {bin: channel for channel, bin in zip(range_excluding_ch10, bins_excluding_ch10)}
        if self.ch10_instrument_bin_name:
            self.bin_channel_map[self.ch10_instrument_bin_name] = 9

    def validate(self):
        if self.max_wait_time % self.wait_events != 0:
            raise ValueError('max_wait_time must be exactly divisible by wait_events')
        if self.velocity_bins < 2:
            raise ValueError('velocity_bins must be at least 2')
        if len(self.bin_instrument_names) > 16:
            raise ValueError('bin_instruments must have at most 16 values')
        if self.velocity_bins_override:
            print('VocabConfig is using velocity_bins_override. Ignoring velocity_exp.')
            if len(self.velocity_bins_override) != self.velocity_bins:
                raise ValueError('velocity_bins_override must have same length as velocity_bins')
        if self.ch10_instrument_bin_name and self.ch10_instrument_bin_name not in self.bin_instrument_names:
            raise ValueError('ch10_instrument_bin_name must be in bin_instruments')
        if self.velocity_exp <= 0:
            raise ValueError('velocity_exp must be greater than 0')

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(**config)