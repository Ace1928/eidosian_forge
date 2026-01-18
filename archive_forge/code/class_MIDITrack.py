from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class MIDITrack(object):
    """
    MIDI Track.

    Parameters
    ----------
    events : list
        MIDI events.

    Notes
    -----
    All events are stored with timing information in absolute ticks.
    The events must be sorted. Consider using `from_notes()` method.

    Examples
    --------

    Create a MIDI track from a list of events. Please note that the events must
    be sorted.

    >>> e1 = NoteOnEvent(tick=100, pitch=50, velocity=60)
    >>> e2 = NoteOffEvent(tick=300, pitch=50)
    >>> e3 = NoteOnEvent(tick=200, pitch=62, velocity=90)
    >>> e4 = NoteOffEvent(tick=600, pitch=62)
    >>> t = MIDITrack(sorted([e1, e2, e3, e4]))
    >>> t  # doctest: +ELLIPSIS
    <madmom.utils.midi.MIDITrack object at 0x...>
    >>> t.events  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [<madmom.utils.midi.NoteOnEvent object at 0x...>,
     <madmom.utils.midi.NoteOnEvent object at 0x...>,
     <madmom.utils.midi.NoteOffEvent object at 0x...>,
     <madmom.utils.midi.NoteOffEvent object at 0x...>]

    It can also be created from an array containing the notes. The `from_notes`
    method also takes care of creating tempo and time signature events.

    >>> notes = np.array([[0.1, 50, 0.3, 60], [0.2, 62, 0.4, 90]])
    >>> t = MIDITrack.from_notes(notes)
    >>> t  # doctest: +ELLIPSIS
    <madmom.utils.midi.MIDITrack object at 0x...>
    >>> t.events  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [<madmom.utils.midi.SetTempoEvent object at 0x...>,
     <madmom.utils.midi.TimeSignatureEvent object at 0...>,
     <madmom.utils.midi.NoteOnEvent object at 0x...>,
     <madmom.utils.midi.NoteOnEvent object at 0x...>,
     <madmom.utils.midi.NoteOffEvent object at 0x...>,
     <madmom.utils.midi.NoteOffEvent object at 0x...>]

    """

    def __init__(self, events=None):
        if events is None:
            self.events = []
        else:
            self.events = events

    def _make_ticks_abs(self):
        """Make the track's events timing information absolute."""
        running_tick = 0
        for event in self.events:
            event.tick += running_tick
            running_tick = event.tick

    def _make_ticks_rel(self):
        """Make the track's events timing information relative."""
        running_tick = 0
        for event in self.events:
            event.tick -= running_tick
            running_tick += event.tick

    @property
    def data_stream(self):
        """
        MIDI data stream representation of the track.

        """
        self.events.sort()
        self._make_ticks_rel()
        status = None
        track_data = bytearray()
        for event in self.events:
            track_data.extend(write_variable_length(event.tick))
            if isinstance(event, MetaEvent):
                track_data.append(event.status_msg)
                track_data.append(event.meta_command)
                track_data.extend(write_variable_length(len(event.data)))
                track_data.extend(event.data)
            elif isinstance(event, SysExEvent):
                track_data.append(240)
                track_data.extend(event.data)
                track_data.append(247)
            elif isinstance(event, Event):
                if not status or status.status_msg != event.status_msg or status.channel != event.channel:
                    status = event
                    track_data.append(event.status_msg | event.channel)
                track_data.extend(event.data)
            else:
                raise ValueError('Unknown MIDI Event: ' + str(event))
        self._make_ticks_abs()
        data = bytearray()
        data.extend(b'MTrk')
        data.extend(struct.pack('>L', len(track_data)))
        data.extend(track_data)
        return data

    @classmethod
    def from_stream(cls, midi_stream):
        """
        Create a MIDI track by reading the data from a stream.

        Parameters
        ----------
        midi_stream : open file handle
            MIDI file stream (e.g. open MIDI file handle)

        Returns
        -------
        :class:`MIDITrack` instance
            :class:`MIDITrack` instance

        """
        events = []
        status = None
        chunk = midi_stream.read(4)
        if chunk != b'MTrk':
            raise TypeError('Bad track header in MIDI file: %s' % chunk)
        track_size = struct.unpack('>L', midi_stream.read(4))[0]
        track_data = iter(midi_stream.read(track_size))
        while True:
            try:
                tick = read_variable_length(track_data)
                status_msg = byte2int(next(track_data))
                if MetaEvent.status_msg == status_msg:
                    meta_cmd = byte2int(next(track_data))
                    if meta_cmd not in EventRegistry.meta_events:
                        import warnings
                        warnings.warn('Unknown Meta MIDI Event: %s' % meta_cmd)
                        event_cls = UnknownMetaEvent
                    else:
                        event_cls = EventRegistry.meta_events[meta_cmd]
                    data_len = read_variable_length(track_data)
                    data = [byte2int(next(track_data)) for _ in range(data_len)]
                    events.append(event_cls(tick=tick, data=data, meta_command=meta_cmd))
                elif SysExEvent.status_msg == status_msg:
                    data = []
                    while True:
                        datum = byte2int(next(track_data))
                        if datum == 247:
                            break
                        data.append(datum)
                    events.append(SysExEvent(tick=tick, data=data))
                else:
                    key = status_msg & 240
                    if key not in EventRegistry.events:
                        assert status, 'Bad byte value'
                        data = []
                        key = status & 240
                        event_cls = EventRegistry.events[key]
                        channel = status & 15
                        data.append(status_msg)
                        data += [byte2int(next(track_data)) for _ in range(event_cls.length - 1)]
                        events.append(event_cls(tick=tick, channel=channel, data=data))
                    else:
                        status = status_msg
                        event_cls = EventRegistry.events[key]
                        channel = status & 15
                        data = [byte2int(next(track_data)) for _ in range(event_cls.length)]
                        events.append(event_cls(tick=tick, channel=channel, data=data))
            except StopIteration:
                break
        track = cls(events)
        track._make_ticks_abs()
        return track

    @classmethod
    def from_notes(cls, notes, tempo=TEMPO, time_signature=TIME_SIGNATURE, resolution=RESOLUTION):
        """
        Create a MIDI track from the given notes.

        Parameters
        ----------
        notes : numpy array
            Array with the notes, one per row. The columns must be:
            (onset time, pitch, duration, velocity, [channel]).
        tempo : float, optional
            Tempo of the MIDI track, given in beats per minute (bpm).
        time_signature : tuple, optional
            Time signature of the track, e.g. (4, 4) for 4/4.
        resolution : int
            Resolution (i.e. ticks per quarter note) of the MIDI track.

        Returns
        -------
        :class:`MIDITrack` instance
            :class:`MIDITrack` instance

        Notes
        -----
        All events including the generated tempo and time signature events is
        included in the returned track (i.e. as defined in MIDI format 0).

        """
        notes = _add_channel(notes)
        sig = TimeSignatureEvent(tick=0)
        sig.numerator, sig.denominator = time_signature
        quarter_note_length = 60.0 / tempo * sig.denominator / 4
        quarter_notes_per_second = 1 / quarter_note_length
        ticks_per_second = resolution * quarter_notes_per_second
        tempo = SetTempoEvent(tick=0)
        tempo.microseconds_per_quarter_note = int(quarter_note_length * 1000000.0)
        events = []
        for note in notes:
            onset, pitch, duration, velocity, channel = note
            e_on = NoteOnEvent()
            e_on.tick = int(onset * ticks_per_second)
            e_on.pitch = int(pitch)
            e_on.velocity = int(velocity)
            e_on.channel = int(channel)
            e_off = NoteOffEvent()
            e_off.tick = int((onset + duration) * ticks_per_second)
            e_off.pitch = int(pitch)
            e_off.channel = int(channel)
            events.append(e_on)
            events.append(e_off)
        events = sorted(events)
        events.insert(0, sig)
        events.insert(0, tempo)
        return cls(events)