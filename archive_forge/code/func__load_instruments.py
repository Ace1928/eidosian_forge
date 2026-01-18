from __future__ import print_function
import mido
import numpy as np
import math
import warnings
import collections
import copy
import functools
import six
from heapq import merge
from .instrument import Instrument
from .containers import (KeySignature, TimeSignature, Lyric, Note,
from .utilities import (key_name_to_key_number, qpm_to_bpm)
def _load_instruments(self, midi_data):
    """Populates ``self.instruments`` using ``midi_data``.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read.
        """
    instrument_map = collections.OrderedDict()
    stragglers = {}
    track_name_map = collections.defaultdict(str)

    def __get_instrument(program, channel, track, create_new):
        """Gets the Instrument corresponding to the given program number,
            drum/non-drum type, channel, and track index.  If no such
            instrument exists, one is created.

            """
        if (program, channel, track) in instrument_map:
            return instrument_map[program, channel, track]
        if not create_new and (channel, track) in stragglers:
            return stragglers[channel, track]
        if create_new:
            is_drum = channel == 9
            instrument = Instrument(program, is_drum, track_name_map[track_idx])
            if (channel, track) in stragglers:
                straggler = stragglers[channel, track]
                instrument.control_changes = straggler.control_changes
                instrument.pitch_bends = straggler.pitch_bends
            instrument_map[program, channel, track] = instrument
        else:
            instrument = Instrument(program, track_name_map[track_idx])
            stragglers[channel, track] = instrument
        return instrument
    for track_idx, track in enumerate(midi_data.tracks):
        last_note_on = collections.defaultdict(list)
        current_instrument = np.zeros(16, dtype=np.int32)
        for event in track:
            if event.type == 'track_name':
                track_name_map[track_idx] = event.name
            if event.type == 'program_change':
                current_instrument[event.channel] = event.program
            elif event.type == 'note_on' and event.velocity > 0:
                note_on_index = (event.channel, event.note)
                last_note_on[note_on_index].append((event.time, event.velocity))
            elif event.type == 'note_off' or (event.type == 'note_on' and event.velocity == 0):
                key = (event.channel, event.note)
                if key in last_note_on:
                    end_tick = event.time
                    open_notes = last_note_on[key]
                    notes_to_close = [(start_tick, velocity) for start_tick, velocity in open_notes if start_tick != end_tick]
                    notes_to_keep = [(start_tick, velocity) for start_tick, velocity in open_notes if start_tick == end_tick]
                    for start_tick, velocity in notes_to_close:
                        start_time = self.__tick_to_time[start_tick]
                        end_time = self.__tick_to_time[end_tick]
                        note = Note(velocity, event.note, start_time, end_time)
                        program = current_instrument[event.channel]
                        instrument = __get_instrument(program, event.channel, track_idx, 1)
                        instrument.notes.append(note)
                    if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
                        last_note_on[key] = notes_to_keep
                    else:
                        del last_note_on[key]
            elif event.type == 'pitchwheel':
                bend = PitchBend(event.pitch, self.__tick_to_time[event.time])
                program = current_instrument[event.channel]
                instrument = __get_instrument(program, event.channel, track_idx, 0)
                instrument.pitch_bends.append(bend)
            elif event.type == 'control_change':
                control_change = ControlChange(event.control, event.value, self.__tick_to_time[event.time])
                program = current_instrument[event.channel]
                instrument = __get_instrument(program, event.channel, track_idx, 0)
                instrument.control_changes.append(control_change)
    self.instruments = [i for i in instrument_map.values()]