from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
class FramedSignal(object):
    """
    The :class:`FramedSignal` splits a :class:`Signal` into frames and makes it
    iterable and indexable.

    Parameters
    ----------
    signal : :class:`Signal` instance
        Signal to be split into frames.
    frame_size : int, optional
        Size of one frame [samples].
    hop_size : float, optional
        Progress `hop_size` samples between adjacent frames.
    fps : float, optional
        Use given frames per second; if set, this computes and overwrites the
        given `hop_size` value.
    origin : int, optional
        Location of the window relative to the reference sample of a frame.
    end : int or str, optional
        End of signal handling (see notes below).
    num_frames : int, optional
        Number of frames to return.
    kwargs : dict, optional
        If no :class:`Signal` instance was given, one is instantiated with
        these additional keyword arguments.

    Notes
    -----
    The :class:`FramedSignal` class is implemented as an iterator. It splits
    the given `signal` automatically into frames of `frame_size` length with
    `hop_size` samples (can be float, normal rounding applies) between the
    frames. The reference sample of the first frame refers to the first sample
    of the `signal`.

    The location of the window relative to the reference sample of a frame can
    be set with the `origin` parameter (with the same behaviour as used by
    ``scipy.ndimage`` filters). Arbitrary integer values can be given:

    - zero centers the window on its reference sample,
    - negative values shift the window to the right,
    - positive values shift the window to the left.

    Additionally, it can have the following literal values:

    - 'center', 'offline': the window is centered on its reference sample,
    - 'left', 'past', 'online': the window is located to the left of its
      reference sample (including the reference sample),
    - 'right', 'future', 'stream': the window is located to the right of its
      reference sample.

    The `end` parameter is used to handle the end of signal behaviour and
    can have these values:

    - 'normal': stop as soon as the whole signal got covered by at least one
      frame (i.e. pad maximally one frame),
    - 'extend': frames are returned as long as part of the frame overlaps
      with the signal to cover the whole signal.

    Alternatively, `num_frames` can be used to retrieve a fixed number of
    frames.

    In order to be able to stack multiple frames obtained with different frame
    sizes, the number of frames to be returned must be independent from the set
    `frame_size`. It is not guaranteed that every sample of the signal is
    returned in a frame unless the `origin` is either 'right' or 'future'.

    If used in online real-time mode the parameters `origin` and `num_frames`
    should be set to 'stream' and 1, respectively.

    Examples
    --------
    To chop a :class:`Signal` (or anything a :class:`Signal` can be
    instantiated from) into overlapping frames of size 2048 with adjacent
    frames being 441 samples apart:

    >>> sig = Signal('tests/data/audio/sample.wav')
    >>> sig
    Signal([-2494, -2510, ...,   655,   639], dtype=int16)
    >>> frames = FramedSignal(sig, frame_size=2048, hop_size=441)
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> frames[0]
    Signal([    0,     0, ..., -4666, -4589], dtype=int16)
    >>> frames[10]
    Signal([-6156, -5645, ...,  -253,   671], dtype=int16)
    >>> frames.fps
    100.0

    Instead of passing a :class:`Signal` instance as the first argument,
    anything a :class:`Signal` can be instantiated from (e.g. a file name) can
    be used. We can also set the frames per second (`fps`) instead, they get
    converted to `hop_size` based on the `sample_rate` of the signal:

    >>> frames = FramedSignal('tests/data/audio/sample.wav', fps=100)
    >>> frames  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>
    >>> frames[0]
    Signal([    0,     0, ..., -4666, -4589], dtype=int16)
    >>> frames.frame_size, frames.hop_size
    (2048, 441.0)

    When trying to access an out of range frame, an IndexError is raised. Thus
    the FramedSignal can be used the same way as a numpy array or any other
    iterable.

    >>> frames = FramedSignal('tests/data/audio/sample.wav')
    >>> frames.num_frames
    281
    >>> frames[281]
    Traceback (most recent call last):
    IndexError: end of signal reached
    >>> frames.shape
    (281, 2048)

    Slices are FramedSignals itself:

    >>> frames[:4]  # doctest: +ELLIPSIS
    <madmom.audio.signal.FramedSignal object at 0x...>

    To obtain a numpy array from a FramedSignal, simply use np.array() on the
    full FramedSignal or a slice of it. Please note, that this requires a full
    memory copy.

    >>> np.array(frames[2:4])
    array([[    0,     0, ..., -5316, -5405],
           [ 2215,  2281, ...,   561,   653]], dtype=int16)

    """

    def __init__(self, signal, frame_size=FRAME_SIZE, hop_size=HOP_SIZE, fps=FPS, origin=ORIGIN, end=END_OF_SIGNAL, num_frames=NUM_FRAMES, **kwargs):
        if not isinstance(signal, Signal):
            signal = Signal(signal, **kwargs)
        self.signal = signal
        if frame_size:
            self.frame_size = int(frame_size)
        if hop_size:
            self.hop_size = float(hop_size)
        if fps:
            self.hop_size = self.signal.sample_rate / float(fps)
        if origin in ('center', 'offline'):
            origin = 0
        elif origin in ('left', 'past', 'online'):
            origin = (frame_size - 1) / 2
        elif origin in ('right', 'future', 'stream'):
            origin = -(frame_size / 2)
        self.origin = int(origin)
        if num_frames is None:
            if end == 'extend':
                num_frames = np.floor(len(self.signal) / float(self.hop_size) + 1)
            elif end == 'normal':
                num_frames = np.ceil(len(self.signal) / float(self.hop_size))
            else:
                raise ValueError("end of signal handling '%s' unknown" % end)
        self.num_frames = int(num_frames)

    def __getitem__(self, index):
        """
        This makes the :class:`FramedSignal` class indexable and/or iterable.

        The signal is split into frames (of length `frame_size`) automatically.
        Two frames are located `hop_size` samples apart. If `hop_size` is a
        float, normal rounding applies.

        """
        if isinstance(index, integer_types):
            if index < 0:
                index += self.num_frames
            if index < self.num_frames:
                return signal_frame(self.signal, index, frame_size=self.frame_size, hop_size=self.hop_size, origin=self.origin)
            raise IndexError('end of signal reached')
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.num_frames)
            if step != 1:
                raise ValueError('only slices with a step size of 1 supported')
            num_frames = stop - start
            origin = self.origin - self.hop_size * start
            return FramedSignal(self.signal, frame_size=self.frame_size, hop_size=self.hop_size, origin=origin, num_frames=num_frames)
        else:
            raise TypeError('frame indices must be slices or integers')

    def __len__(self):
        return self.num_frames

    @property
    def frame_rate(self):
        """Frame rate (same as fps)."""
        if self.signal.sample_rate is None:
            return None
        return float(self.signal.sample_rate) / self.hop_size

    @property
    def fps(self):
        """Frames per second."""
        return self.frame_rate

    @property
    def overlap_factor(self):
        """Overlapping factor of two adjacent frames."""
        return 1.0 - self.hop_size / self.frame_size

    @property
    def shape(self):
        """
        Shape of the FramedSignal (num_frames, frame_size[, num_channels]).

        """
        shape = (self.num_frames, self.frame_size)
        if self.signal.num_channels != 1:
            shape += (self.signal.num_channels,)
        return shape

    @property
    def ndim(self):
        """Dimensionality of the FramedSignal."""
        return len(self.shape)

    def energy(self):
        """Energy of the individual frames."""
        return energy(self)

    def root_mean_square(self):
        """Root mean square of the individual frames."""
        return root_mean_square(self)
    rms = root_mean_square

    def sound_pressure_level(self):
        """Sound pressure level of the individual frames."""
        return sound_pressure_level(self)
    spl = sound_pressure_level