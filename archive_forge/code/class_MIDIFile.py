from __future__ import absolute_import, division, print_function
import numpy as np
import mido
class MIDIFile(mido.MidiFile):
    """
    MIDI File.

    Parameters
    ----------
    filename : str
        MIDI file name.
    file_format : int, optional
        MIDI file format (0, 1, 2).
    ticks_per_beat : int, optional
        Resolution (i.e. ticks per quarter note) of the MIDI file.
    unit : str, optional
        Unit of all MIDI messages, can be one of the following:

        - 'ticks', 't': use native MIDI ticks as unit,
        - 'seconds', 's': use seconds as unit,
        - 'beats', 'b' : use beats as unit.

    timing : str, optional
        Timing of all MIDI messages, can be one of the following:

        - 'absolute', 'abs', 'a': use absolute timing.
        - 'relative', 'rel', 'r': use relative timing, i.e. delta to
        previous message.

    Examples
    --------
    Create a MIDI file from an array with notes. The format of the note array
    is: 'onset time', 'pitch', 'duration', 'velocity', 'channel'. The last
    column can be omitted, assuming channel 0.

    >>> notes = np.array([[0, 50, 1, 60], [0.5, 62, 0.5, 90]])
    >>> m = MIDIFile.from_notes(notes)
    >>> m  # doctest: +ELLIPSIS
    <madmom.io.midi.MIDIFile object at 0x...>

    The notes can be accessed as a numpy array in various formats (default is
    seconds):

    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit ='ticks'
    >>> m.notes
    array([[  0.,  50., 960.,  60.,   0.],
           [480.,  62., 480.,  90.,   0.]])
    >>> m.unit = 'seconds'
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0., 50.,  2., 60.,  0.],
           [ 1., 62.,  1., 90.,  0.]])

    >>> m = MIDIFile.from_notes(notes, tempo=60)
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[  0.,  50., 480.,  60.,   0.],
           [240.,  62., 240.,  90.,   0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])

    >>> m = MIDIFile.from_notes(notes, time_signature=(2, 2))
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[   0.,   50., 1920.,   60.,    0.],
           [ 960.,   62.,  960.,   90.,    0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0., 50.,  2., 60.,  0.],
           [ 1., 62.,  1., 90.,  0.]])

    >>> m = MIDIFile.from_notes(notes, tempo=60, time_signature=(2, 2))
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[  0.,  50., 960.,  60.,   0.],
           [480.,  62., 480.,  90.,   0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])

    >>> m = MIDIFile.from_notes(notes, tempo=240, time_signature=(3, 8))
    >>> m.notes
    array([[ 0. , 50. ,  1. , 60. ,  0. ],
           [ 0.5, 62. ,  0.5, 90. ,  0. ]])
    >>> m.unit = 'ticks'
    >>> m.notes
    array([[  0.,  50., 960.,  60.,   0.],
           [480.,  62., 480.,  90.,   0.]])
    >>> m.unit = 'beats'
    >>> m.notes
    array([[ 0., 50.,  4., 60.,  0.],
           [ 2., 62.,  2., 90.,  0.]])

    """
    UNIT = 'seconds'
    TIMING = 'absolute'

    def __init__(self, filename=None, file_format=0, ticks_per_beat=DEFAULT_TICKS_PER_BEAT, unit=UNIT, timing=TIMING, **kwargs):
        super(MIDIFile, self).__init__(filename=filename, type=file_format, ticks_per_beat=ticks_per_beat, **kwargs)
        self.unit = unit
        self.timing = timing

    def __iter__(self):
        if self.type == 2:
            raise TypeError("can't merge tracks in type 2 (asynchronous) file")
        tempo = DEFAULT_TEMPO
        time_signature = DEFAULT_TIME_SIGNATURE
        cum_delta = 0
        for msg in mido.merge_tracks(self.tracks):
            if msg.time > 0:
                if self.unit.lower() in ('t', 'ticks'):
                    delta = msg.time
                elif self.unit.lower() in ('s', 'sec', 'seconds'):
                    delta = tick2second(msg.time, self.ticks_per_beat, tempo)
                elif self.unit.lower() in ('b', 'beats'):
                    delta = tick2beat(msg.time, self.ticks_per_beat, time_signature)
                else:
                    raise ValueError("`unit` must be either 'ticks', 't', 'seconds', 's', 'beats', 'b', not %s." % self.unit)
            else:
                delta = 0
            if self.timing.lower() in ('a', 'abs', 'absolute'):
                cum_delta += delta
            elif self.timing.lower() in ('r', 'rel', 'relative'):
                cum_delta = delta
            else:
                raise ValueError("`timing` must be either 'relative', 'rel', 'r', or 'absolute', 'abs', 'a', not %s." % self.timing)
            yield msg.copy(time=cum_delta)
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'time_signature':
                time_signature = (msg.numerator, msg.denominator)

    def __repr__(self):
        return object.__repr__(self)

    @property
    def tempi(self):
        """
        Tempi (mircoseconds per quarter note) of the MIDI file.

        Returns
        -------
        tempi : numpy array
            Array with tempi (time, tempo).

        Notes
        -----
        The time will be given in the unit set by `unit`.

        """
        tempi = []
        for msg in self:
            if msg.type == 'set_tempo':
                tempi.append((msg.time, msg.tempo))
        if not tempi or tempi[0][0] > 0:
            tempi.insert(0, (0, DEFAULT_TEMPO))
        return np.asarray(tempi, np.float)

    @property
    def time_signatures(self):
        """
        Time signatures of the MIDI file.

        Returns
        -------
        time_signatures : numpy array
            Array with time signatures (time, numerator, denominator).

        Notes
        -----
        The time will be given in the unit set by `unit`.

        """
        signatures = []
        for msg in self:
            if msg.type == 'time_signature':
                signatures.append((msg.time, msg.numerator, msg.denominator))
        if not signatures or signatures[0][0] > 0:
            signatures.insert(0, (0, DEFAULT_TIME_SIGNATURE[0], DEFAULT_TIME_SIGNATURE[1]))
        return np.asarray(signatures, dtype=np.float)

    @property
    def notes(self):
        """
        Notes of the MIDI file.

        Returns
        -------
        notes : numpy array
            Array with notes (onset time, pitch, duration, velocity, channel).

        """
        notes = []
        sounding_notes = {}

        def note_hash(channel, pitch):
            """Generate a note hash."""
            return channel * 128 + pitch
        for msg in self:
            note_on = msg.type == 'note_on'
            note_off = msg.type == 'note_off'
            if note_on or note_off:
                note = note_hash(msg.channel, msg.note)
            if note_on and msg.velocity > 0:
                sounding_notes[note] = (msg.time, msg.velocity)
            elif note_off or (note_on and msg.velocity == 0):
                if note not in sounding_notes:
                    import warnings
                    warnings.warn('ignoring MIDI message %s' % msg)
                    continue
                notes.append((sounding_notes[note][0], msg.note, msg.time - sounding_notes[note][0], sounding_notes[note][1], msg.channel))
                del sounding_notes[note]
        return np.asarray(sorted(notes), dtype=np.float)

    @classmethod
    def from_notes(cls, notes, unit='seconds', tempo=DEFAULT_TEMPO, time_signature=DEFAULT_TIME_SIGNATURE, ticks_per_beat=DEFAULT_TICKS_PER_BEAT):
        """
        Create a MIDIFile from the given notes.

        Parameters
        ----------
        notes : numpy array
            Array with notes, one per row. The columns are defined as:
            (onset time, pitch, duration, velocity, [channel]).
        unit : str, optional
            Unit of `notes`, can be one of the following:

            - 'seconds', 's': use seconds as unit,
            - 'ticks', 't': use native MIDI ticks as unit,
            - 'beats', 'b' : use beats as unit.

        tempo : float, optional
            Tempo of the MIDI track, given in bpm or microseconds per quarter
            note. The unit is determined automatically by the value:

            - `tempo` <= 1000: bpm
            - `tempo` > 1000: microseconds per quarter note

        time_signature : tuple, optional
            Time signature of the track, e.g. (4, 4) for 4/4.
        ticks_per_beat : int, optional
            Resolution (i.e. ticks per quarter note) of the MIDI file.

        Returns
        -------
        :class:`MIDIFile` instance
            :class:`MIDIFile` instance with all notes collected in one track.

        Notes
        -----
        All note events (including the generated tempo and time signature
        events) are written into a single track (i.e. MIDI file format 0).

        """
        midi_file = cls(file_format=0, ticks_per_beat=ticks_per_beat, unit=unit, timing='absolute')
        if tempo <= 1000:
            tempo = bpm2tempo(tempo, time_signature)
        else:
            tempo = int(tempo * time_signature[1] / 4)
        track = midi_file.add_track()
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        track.append(mido.MetaMessage('time_signature', numerator=time_signature[0], denominator=time_signature[1]))
        messages = []
        for note in notes:
            try:
                onset, pitch, duration, velocity, channel = note
                channel = int(channel)
                velocity = int(velocity)
            except ValueError:
                onset, pitch, duration, velocity = note
                channel = 0
            pitch = int(pitch)
            velocity = int(velocity)
            offset = onset + duration
            onset = second2tick(onset, ticks_per_beat, tempo)
            note_on = mido.Message('note_on', time=onset, note=pitch, velocity=velocity, channel=channel)
            offset = second2tick(offset, ticks_per_beat, tempo)
            note_off = mido.Message('note_off', time=offset, note=pitch, channel=channel)
            messages.extend([note_on, note_off])
        messages.sort(key=lambda msg: msg.time)
        messages = mido.midifiles.tracks._to_reltime(messages)
        track.extend(messages)
        return midi_file

    def save(self, filename):
        """
        Save to MIDI file.

        Parameters
        ----------
        filename : str or open file handle
            The MIDI file name.

        """
        from . import open_file
        with open_file(filename, 'wb') as f:
            self._save(f)