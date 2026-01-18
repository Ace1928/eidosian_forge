import numpy as np
import os
import pkg_resources
from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz
class Instrument(object):
    """Object to hold event information for a single instrument.

    Parameters
    ----------
    program : int
        MIDI program number (instrument index), in ``[0, 127]``.
    is_drum : bool
        Is the instrument a drum instrument (channel 9)?
    name : str
        Name of the instrument.

    Attributes
    ----------
    program : int
        The program number of this instrument.
    is_drum : bool
        Is the instrument a drum instrument (channel 9)?
    name : str
        Name of the instrument.
    notes : list
        List of :class:`pretty_midi.Note` objects.
    pitch_bends : list
        List of of :class:`pretty_midi.PitchBend` objects.
    control_changes : list
        List of :class:`pretty_midi.ControlChange` objects.

    """

    def __init__(self, program, is_drum=False, name=''):
        """Create the Instrument.

        """
        self.program = program
        self.is_drum = is_drum
        self.name = name
        self.notes = []
        self.pitch_bends = []
        self.control_changes = []

    def get_onsets(self):
        """Get all onsets of all notes played by this instrument.
        May contain duplicates.

        Returns
        -------
        onsets : np.ndarray
                List of all note onsets.

        """
        onsets = []
        for note in self.notes:
            onsets.append(note.start)
        return np.sort(onsets)

    def get_piano_roll(self, fs=100, times=None, pedal_threshold=64):
        """Compute a piano roll matrix of this instrument.

        Parameters
        ----------
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart
            by ``1./fs`` seconds.
        times : np.ndarray
            Times of the start of each column in the piano roll.
            Default ``None`` which is ``np.arange(0, get_end_time(), 1./fs)``.
        pedal_threshold : int
            Value of control change 64 (sustain pedal) message that is less
            than this value is reflected as pedal-off.  Pedals will be
            reflected as elongation of notes in the piano roll.
            If None, then CC64 message is ignored.
            Default is 64.

        Returns
        -------
        piano_roll : np.ndarray, shape=(128,times.shape[0])
            Piano roll of this instrument.

        """
        if self.notes == []:
            return np.array([[]] * 128)
        end_time = self.get_end_time()
        if times is not None and times[-1] > end_time:
            end_time = times[-1]
        piano_roll = np.zeros((128, int(fs * end_time)))
        if self.is_drum:
            if times is None:
                return piano_roll
            else:
                return np.zeros((128, times.shape[0]))
        for note in self.notes:
            piano_roll[note.pitch, int(note.start * fs):int(note.end * fs)] += note.velocity
        if pedal_threshold is not None:
            CC_SUSTAIN_PEDAL = 64
            time_pedal_on = 0
            is_pedal_on = False
            for cc in [_e for _e in self.control_changes if _e.number == CC_SUSTAIN_PEDAL]:
                time_now = int(cc.time * fs)
                is_current_pedal_on = cc.value >= pedal_threshold
                if not is_pedal_on and is_current_pedal_on:
                    time_pedal_on = time_now
                    is_pedal_on = True
                elif is_pedal_on and (not is_current_pedal_on):
                    subpr = piano_roll[:, time_pedal_on:time_now]
                    pedaled = np.maximum.accumulate(subpr, axis=1)
                    piano_roll[:, time_pedal_on:time_now] = pedaled
                    is_pedal_on = False
        ordered_bends = sorted(self.pitch_bends, key=lambda bend: bend.time)
        end_bend = PitchBend(0, end_time)
        for start_bend, end_bend in zip(ordered_bends, ordered_bends[1:] + [end_bend]):
            if np.abs(start_bend.pitch) < 1:
                continue
            start_pitch = pitch_bend_to_semitones(start_bend.pitch)
            bend_int = int(np.sign(start_pitch) * np.floor(np.abs(start_pitch)))
            bend_decimal = np.abs(start_pitch - bend_int)
            bend_range = np.r_[int(start_bend.time * fs):int(end_bend.time * fs)]
            bent_roll = np.zeros(piano_roll[:, bend_range].shape)
            if start_bend.pitch >= 0:
                if bend_int != 0:
                    bent_roll[bend_int:] = piano_roll[:-bend_int, bend_range]
                else:
                    bent_roll = piano_roll[:, bend_range]
                bent_roll[1:] = (1 - bend_decimal) * bent_roll[1:] + bend_decimal * bent_roll[:-1]
            else:
                if bend_int != 0:
                    bent_roll[:bend_int] = piano_roll[-bend_int:, bend_range]
                else:
                    bent_roll = piano_roll[:, bend_range]
                bent_roll[:-1] = (1 - bend_decimal) * bent_roll[:-1] + bend_decimal * bent_roll[1:]
            piano_roll[:, bend_range] = bent_roll
        if times is None:
            return piano_roll
        piano_roll_integrated = np.zeros((128, times.shape[0]))
        times = np.array(np.round(times * fs), dtype=np.int32)
        for n, (start, end) in enumerate(zip(times[:-1], times[1:])):
            if start < piano_roll.shape[1]:
                if start == end:
                    end = start + 1
                piano_roll_integrated[:, n] = np.mean(piano_roll[:, start:end], axis=1)
        return piano_roll_integrated

    def get_chroma(self, fs=100, times=None, pedal_threshold=64):
        """Get a sequence of chroma vectors from this instrument.

        Parameters
        ----------
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart
            by ``1./fs`` seconds.
        times : np.ndarray
            Times of the start of each column in the piano roll.
            Default ``None`` which is ``np.arange(0, get_end_time(), 1./fs)``.
        pedal_threshold : int
            Value of control change 64 (sustain pedal) message that is less
            than this value is reflected as pedal-off.  Pedals will be
            reflected as elongation of notes in the piano roll.
            If None, then CC64 message is ignored.
            Default is 64.

        Returns
        -------
        piano_roll : np.ndarray, shape=(12,times.shape[0])
            Chromagram of this instrument.

        """
        piano_roll = self.get_piano_roll(fs=fs, times=times, pedal_threshold=pedal_threshold)
        chroma_matrix = np.zeros((12, piano_roll.shape[1]))
        for note in range(12):
            chroma_matrix[note, :] = np.sum(piano_roll[note::12], axis=0)
        return chroma_matrix

    def get_end_time(self):
        """Returns the time of the end of the events in this instrument.

        Returns
        -------
        end_time : float
            Time, in seconds, of the last event.

        """
        events = [n.end for n in self.notes] + [b.time for b in self.pitch_bends] + [c.time for c in self.control_changes]
        if len(events) == 0:
            return 0.0
        else:
            return max(events)

    def get_pitch_class_histogram(self, use_duration=False, use_velocity=False, normalize=False):
        """Computes the frequency of pitch classes of this instrument,
        optionally weighted by their durations or velocities.

        Parameters
        ----------
        use_duration : bool
            Weight frequency by note duration.
        use_velocity : bool
            Weight frequency by note velocity.
        normalize : bool
            Normalizes the histogram such that the sum of bin values is 1.

        Returns
        -------
        histogram : np.ndarray, shape=(12,)
            Histogram of pitch classes given current instrument, optionally
            weighted by their durations or velocities.
        """
        if self.is_drum:
            return np.zeros(12)
        weights = np.ones(len(self.notes))
        if use_duration:
            weights *= [note.end - note.start for note in self.notes]
        if use_velocity:
            weights *= [note.velocity for note in self.notes]
        histogram, _ = np.histogram([n.pitch % 12 for n in self.notes], bins=np.arange(13), weights=weights, density=normalize)
        return histogram

    def get_pitch_class_transition_matrix(self, normalize=False, time_thresh=0.05):
        """Computes the pitch class transition matrix of this instrument.
        Transitions are added whenever the end of a note is within
        ``time_tresh`` from the start of any other note.

        Parameters
        ----------
        normalize : bool
            Normalize transition matrix such that matrix sum equals to 1.
        time_thresh : float
            Maximum temporal threshold, in seconds, between the start of a note
            and end time of any other note for a transition to be added.

        Returns
        -------
        transition_matrix : np.ndarray, shape=(12,12)
            Pitch class transition matrix.
        """
        if self.is_drum or len(self.notes) <= 1:
            return np.zeros((12, 12))
        starts, ends, nodes = np.array([[x.start, x.end, x.pitch % 12] for x in self.notes]).T
        dist_mat = np.subtract.outer(ends, starts)
        sources, targets = np.where(abs(dist_mat) < time_thresh)
        transition_matrix, _, _ = np.histogram2d(nodes[sources], nodes[targets], bins=np.arange(13), normed=normalize)
        return transition_matrix

    def remove_invalid_notes(self):
        """Removes any notes whose end time is before or at their start time.

        """
        notes_to_delete = []
        for note in self.notes:
            if note.end <= note.start:
                notes_to_delete.append(note)
        for note in notes_to_delete:
            self.notes.remove(note)

    def synthesize(self, fs=44100, wave=np.sin):
        """Synthesize the instrument's notes using some waveshape.
        For drum instruments, returns zeros.

        Parameters
        ----------
        fs : int
            Sampling rate of the synthesized audio signal.
        wave : function
            Function which returns a periodic waveform,
            e.g. ``np.sin``, ``scipy.signal.square``, etc.

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the instrument's notes, synthesized at ``fs``.

        """
        synthesized = np.zeros(int(fs * (self.get_end_time() + 1)))
        if self.is_drum:
            return synthesized
        if not hasattr(wave, '__call__'):
            raise ValueError('wave should be a callable Python function')
        fade_out = np.linspace(1, 0, int(0.1 * fs))
        bend_multiplier = np.ones(synthesized.shape)
        ordered_bends = sorted(self.pitch_bends, key=lambda bend: bend.time)
        end_bend = PitchBend(0, self.get_end_time())
        for start_bend, end_bend in zip(ordered_bends, ordered_bends[1:] + [end_bend]):
            start = int(start_bend.time * fs)
            end = int(end_bend.time * fs)
            bend_semitones = pitch_bend_to_semitones(start_bend.pitch)
            bend_amount = (2 ** (1 / 12.0)) ** bend_semitones
            bend_multiplier[start:end] = bend_amount
        for note in self.notes:
            start = int(fs * note.start)
            end = int(fs * note.end)
            frequency = note_number_to_hz(note.pitch)
            offsets = np.zeros(end - start)
            for bend in ordered_bends:
                bend_sample = int(bend.time * fs)
                if bend_sample > start and bend_sample < end:
                    bend_so_far = bend_multiplier[start:bend_sample].mean()
                    bend_amount = bend_multiplier[bend_sample]
                    offset = (bend_so_far - bend_amount) * (bend_sample - start)
                    offsets[bend_sample - start:] = offset
            frequencies = 2 * np.pi * frequency * bend_multiplier[start:end] / fs
            note_waveform = wave(frequencies * np.arange(end - start) + 2 * np.pi * frequency * offsets / fs)
            envelope = np.exp(-np.arange(end - start) / (1.0 * fs))
            if envelope.shape[0] > fade_out.shape[0]:
                envelope[-fade_out.shape[0]:] *= fade_out
            else:
                envelope *= np.linspace(1, 0, envelope.shape[0])
            envelope *= note.velocity
            synthesized[start:end] += envelope * note_waveform
        return synthesized

    def fluidsynth(self, fs=44100, sf2_path=None):
        """Synthesize using fluidsynth.

        Parameters
        ----------
        fs : int
            Sampling rate to synthesize.
        sf2_path : str
            Path to a .sf2 file.
            Default ``None``, which uses the TimGM6mb.sf2 file included with
            ``pretty_midi``.

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at ``fs``.

        """
        if sf2_path is None:
            sf2_path = pkg_resources.resource_filename(__name__, DEFAULT_SF2)
        if not _HAS_FLUIDSYNTH:
            raise ImportError('fluidsynth() was called but pyfluidsynth is not installed.')
        if not os.path.exists(sf2_path):
            raise ValueError('No soundfont file found at the supplied path {}'.format(sf2_path))
        if len(self.notes) == 0:
            return np.array([])
        fl = fluidsynth.Synth(samplerate=fs)
        sfid = fl.sfload(sf2_path)
        if self.is_drum:
            channel = 9
            res = fl.program_select(channel, sfid, 128, self.program)
            if res == -1:
                fl.program_select(channel, sfid, 128, 0)
        else:
            channel = 0
            fl.program_select(channel, sfid, 0, self.program)
        event_list = []
        for note in self.notes:
            event_list += [[note.start, 'note on', note.pitch, note.velocity]]
            event_list += [[note.end, 'note off', note.pitch]]
        for bend in self.pitch_bends:
            event_list += [[bend.time, 'pitch bend', bend.pitch]]
        for control_change in self.control_changes:
            event_list += [[control_change.time, 'control change', control_change.number, control_change.value]]
        event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
        current_time = event_list[0][0]
        next_event_times = [e[0] for e in event_list[1:]]
        for event, end in zip(event_list[:-1], next_event_times):
            event[0] = end - event[0]
        event_list[-1][0] = 1.0
        total_time = current_time + np.sum([e[0] for e in event_list])
        synthesized = np.zeros(int(np.ceil(fs * total_time)))
        for event in event_list:
            if event[1] == 'note on':
                fl.noteon(channel, event[2], event[3])
            elif event[1] == 'note off':
                fl.noteoff(channel, event[2])
            elif event[1] == 'pitch bend':
                fl.pitch_bend(channel, event[2])
            elif event[1] == 'control change':
                fl.cc(channel, event[2], event[3])
            current_sample = int(fs * current_time)
            end = int(fs * (current_time + event[0]))
            samples = fl.get_samples(end - current_sample)[::2]
            synthesized[current_sample:end] += samples
            current_time += event[0]
        fl.delete()
        return synthesized

    def __repr__(self):
        return 'Instrument(program={}, is_drum={}, name="{}")'.format(self.program, self.is_drum, self.name.replace('"', '\\"'))