from __future__ import division
import logging
import os
import re
import subprocess as sp
import warnings
import numpy as np
from moviepy.compat import DEVNULL, PY3
from moviepy.config import get_setting  # ffmpeg, ffmpeg.exe, etc...
from moviepy.tools import cvsecs
class FFMPEG_VideoReader:

    def __init__(self, filename, print_infos=False, bufsize=None, pix_fmt='rgb24', check_duration=True, target_resolution=None, resize_algo='bicubic', fps_source='tbr'):
        self.filename = filename
        self.proc = None
        infos = ffmpeg_parse_infos(filename, print_infos, check_duration, fps_source)
        self.fps = infos['video_fps']
        self.size = infos['video_size']
        self.rotation = infos['video_rotation']
        if target_resolution:
            target_resolution = (target_resolution[1], target_resolution[0])
            if None in target_resolution:
                ratio = 1
                for idx, target in enumerate(target_resolution):
                    if target:
                        ratio = target / self.size[idx]
                self.size = (int(self.size[0] * ratio), int(self.size[1] * ratio))
            else:
                self.size = target_resolution
        self.resize_algo = resize_algo
        self.duration = infos['video_duration']
        self.ffmpeg_duration = infos['duration']
        self.nframes = infos['video_nframes']
        self.infos = infos
        self.pix_fmt = pix_fmt
        self.depth = 4 if pix_fmt == 'rgba' else 3
        if bufsize is None:
            w, h = self.size
            bufsize = self.depth * w * h + 100
        self.bufsize = bufsize
        self.initialize()
        self.pos = 1
        self.lastread = self.read_frame()

    def initialize(self, starttime=0):
        """Opens the file, creates the pipe. """
        self.close()
        if starttime != 0:
            offset = min(1, starttime)
            i_arg = ['-ss', '%.06f' % (starttime - offset), '-i', self.filename, '-ss', '%.06f' % offset]
        else:
            i_arg = ['-i', self.filename]
        cmd = [get_setting('FFMPEG_BINARY')] + i_arg + ['-loglevel', 'error', '-f', 'image2pipe', '-vf', 'scale=%d:%d' % tuple(self.size), '-sws_flags', self.resize_algo, '-pix_fmt', self.pix_fmt, '-vcodec', 'rawvideo', '-']
        popen_params = {'bufsize': self.bufsize, 'stdout': sp.PIPE, 'stderr': sp.PIPE, 'stdin': DEVNULL}
        if os.name == 'nt':
            popen_params['creationflags'] = 134217728
        self.proc = sp.Popen(cmd, **popen_params)

    def skip_frames(self, n=1):
        """Reads and throws away n frames """
        w, h = self.size
        for i in range(n):
            self.proc.stdout.read(self.depth * w * h)
        self.pos += n

    def read_frame(self):
        w, h = self.size
        nbytes = self.depth * w * h
        s = self.proc.stdout.read(nbytes)
        if len(s) != nbytes:
            warnings.warn('Warning: in file %s, ' % self.filename + '%d bytes wanted but %d bytes read,' % (nbytes, len(s)) + 'at frame %d/%d, at time %.02f/%.02f sec. ' % (self.pos, self.nframes, 1.0 * self.pos / self.fps, self.duration) + 'Using the last valid frame instead.', UserWarning)
            if not hasattr(self, 'lastread'):
                raise IOError('MoviePy error: failed to read the first frame of video file %s. That might mean that the file is corrupted. That may also mean that you are using a deprecated version of FFMPEG. On Ubuntu/Debian for instance the version in the repos is deprecated. Please update to a recent version from the website.' % self.filename)
            result = self.lastread
        else:
            if hasattr(np, 'frombuffer'):
                result = np.frombuffer(s, dtype='uint8')
            else:
                result = np.fromstring(s, dtype='uint8')
            result.shape = (h, w, len(s) // (w * h))
            self.lastread = result
        return result

    def get_frame(self, t):
        """ Read a file video frame at time t.

        Note for coders: getting an arbitrary frame in the video with
        ffmpeg can be painfully slow if some decoding has to be done.
        This function tries to avoid fetching arbitrary frames
        whenever possible, by moving between adjacent frames.
        """
        pos = int(self.fps * t + 1e-05) + 1
        if not self.proc:
            self.initialize(t)
            self.pos = pos
            self.lastread = self.read_frame()
        if pos == self.pos:
            return self.lastread
        elif pos < self.pos or pos > self.pos + 100:
            self.initialize(t)
            self.pos = pos
        else:
            self.skip_frames(pos - self.pos - 1)
        result = self.read_frame()
        self.pos = pos
        return result

    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc.stdout.close()
            self.proc.stderr.close()
            self.proc.wait()
            self.proc = None
        if hasattr(self, 'lastread'):
            del self.lastread

    def __del__(self):
        self.close()