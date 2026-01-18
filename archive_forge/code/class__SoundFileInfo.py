import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
class _SoundFileInfo(object):
    """Information about a SoundFile"""

    def __init__(self, file, verbose):
        self.verbose = verbose
        with SoundFile(file) as f:
            self.name = f.name
            self.samplerate = f.samplerate
            self.channels = f.channels
            self.frames = f.frames
            self.duration = float(self.frames) / f.samplerate
            self.format = f.format
            self.subtype = f.subtype
            self.endian = f.endian
            self.format_info = f.format_info
            self.subtype_info = f.subtype_info
            self.sections = f.sections
            self.extra_info = f.extra_info

    @property
    def _duration_str(self):
        hours, rest = divmod(self.duration, 3600)
        minutes, seconds = divmod(rest, 60)
        if hours >= 1:
            duration = '{0:.0g}:{1:02.0g}:{2:05.3f} h'.format(hours, minutes, seconds)
        elif minutes >= 1:
            duration = '{0:02.0g}:{1:05.3f} min'.format(minutes, seconds)
        elif seconds <= 1:
            duration = '{0:d} samples'.format(self.frames)
        else:
            duration = '{0:.3f} s'.format(seconds)
        return duration

    def __repr__(self):
        info = '\n'.join(['{0.name}', 'samplerate: {0.samplerate} Hz', 'channels: {0.channels}', 'duration: {0._duration_str}', 'format: {0.format_info} [{0.format}]', 'subtype: {0.subtype_info} [{0.subtype}]'])
        if self.verbose:
            info += '\n'.join(['\nendian: {0.endian}', 'sections: {0.sections}', 'frames: {0.frames}', 'extra_info: """', '    {1}"""'])
        indented_extra_info = ('\n' + ' ' * 4).join(self.extra_info.split('\n'))
        return info.format(self, indented_extra_info)