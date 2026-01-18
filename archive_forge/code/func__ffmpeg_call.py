from __future__ import absolute_import, division, print_function
import errno
import os
import subprocess
import sys
import tempfile
import numpy as np
from ..utils import string_types
from ..audio.signal import Signal
def _ffmpeg_call(infile, output, fmt='f32le', sample_rate=None, num_channels=1, skip=None, max_len=None, cmd='ffmpeg'):
    """
    Create a sequence of strings indicating ffmpeg how to be called as well as
    the parameters necessary to decode the given input (file) to the given
    format, at the given offset and for the given length to the given output.

    Parameters
    ----------
    infile : str
        Name of the audio sound file to decode.
    output : str
        Where to decode to.
    fmt : {'f32le', 's16le'}, optional
        Format of the samples:
        - 'f32le' for float32, little-endian,
        - 's16le' for signed 16-bit int, little-endian.
    sample_rate : int, optional
        Sample rate to re-sample the signal to (if set) [Hz].
    num_channels : int, optional
        Number of channels to reduce the signal to.
    skip : float, optional
        Number of seconds to skip at beginning of file.
    max_len : float, optional
        Maximum length in seconds to decode.
    cmd : {'ffmpeg','avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).

    Returns
    -------
    list
        ffmpeg call.

    Notes
    -----
    'avconv' rounds decoding positions and decodes in blocks of 4096 length
    resulting in incorrect start and stop positions. Thus it should only be
    used to decode complete files.

    """
    if cmd == 'avconv' and skip is not None and (max_len is not None):
        raise RuntimeError('avconv has a bug, which results in wrong audio slices! Decode the audio files to .wav first or use ffmpeg.')
    if isinstance(infile, Signal):
        in_fmt = _ffmpeg_fmt(infile.dtype)
        in_ac = str(int(infile.num_channels))
        in_ar = str(int(infile.sample_rate))
        infile = str('pipe:0')
    else:
        infile = str(infile)
    call = [cmd, '-v', 'quiet', '-y']
    if skip:
        call.extend(['-ss', '%f' % float(skip)])
    if infile == 'pipe:0':
        call.extend(['-f', in_fmt, '-ac', in_ac, '-ar', in_ar])
    call.extend(['-i', infile])
    call.extend(['-f', str(fmt)])
    if max_len:
        call.extend(['-t', '%f' % float(max_len)])
    if num_channels:
        call.extend(['-ac', str(int(num_channels))])
    if sample_rate:
        call.extend(['-ar', str(int(sample_rate))])
    call.append(output)
    return call