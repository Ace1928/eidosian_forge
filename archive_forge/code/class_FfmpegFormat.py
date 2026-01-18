import re
import sys
import time
import logging
import platform
import threading
import subprocess as sp
import imageio_ffmpeg
import numpy as np
from ..core import Format, image_as_uint
class FfmpegFormat(Format):
    """Read/Write ImageResources using FFMPEG.

    See :mod:`imageio.plugins.ffmpeg`
    """

    def _can_read(self, request):
        if re.match('<video(\\d+)>', request.filename):
            return True
        if request.extension in self.extensions:
            return True

    def _can_write(self, request):
        if request.extension in self.extensions:
            return True

    class Reader(Format.Reader):
        _frame_catcher = None
        _read_gen = None

        def _get_cam_inputname(self, index):
            if sys.platform.startswith('linux'):
                return '/dev/' + self.request._video[1:-1]
            elif sys.platform.startswith('win'):
                ffmpeg_api = imageio_ffmpeg
                cmd = [ffmpeg_api.get_ffmpeg_exe(), '-list_devices', 'true', '-f', CAM_FORMAT, '-i', 'dummy']
                completed_process = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf-8', shell=True, check=False)
                try:
                    name = parse_device_names(completed_process.stderr)[index]
                except IndexError:
                    raise IndexError('No ffdshow camera at index %i.' % index)
                return 'video=%s' % name
            elif sys.platform.startswith('darwin'):
                name = str(index)
                return name
            else:
                return '??'

        def _open(self, loop=False, size=None, dtype=None, pixelformat=None, print_info=False, ffmpeg_params=None, input_params=None, output_params=None, fps=None):
            self._ffmpeg_api = imageio_ffmpeg
            self._arg_loop = bool(loop)
            if size is None:
                self._arg_size = None
            elif isinstance(size, tuple):
                self._arg_size = '%ix%i' % size
            elif isinstance(size, str) and 'x' in size:
                self._arg_size = size
            else:
                raise ValueError('FFMPEG size must be tuple of "NxM"')
            if pixelformat is None:
                pass
            elif not isinstance(pixelformat, str):
                raise ValueError('FFMPEG pixelformat must be str')
            if dtype is None:
                self._dtype = np.dtype('uint8')
            else:
                self._dtype = np.dtype(dtype)
                allowed_dtypes = ['uint8', 'uint16']
                if self._dtype.name not in allowed_dtypes:
                    raise ValueError('dtype must be one of: {}'.format(', '.join(allowed_dtypes)))
            self._arg_pixelformat = pixelformat
            self._arg_input_params = input_params or []
            self._arg_output_params = output_params or []
            self._arg_input_params += ffmpeg_params or []
            self.request._video = None
            regex_match = re.match('<video(\\d+)>', self.request.filename)
            if regex_match:
                self.request._video = self.request.filename
            if self.request._video:
                index = int(regex_match.group(1))
                self._filename = self._get_cam_inputname(index)
            else:
                self._filename = self.request.get_local_filename()
                self._filename = self._filename.replace('^', '^^')
            self._depth = 3
            if self._dtype.name == 'uint8':
                self._pix_fmt = 'rgb24'
                self._bytes_per_channel = 1
            else:
                self._pix_fmt = 'rgb48le'
                self._bytes_per_channel = 2
            self._pos = -1
            self._meta = {'plugin': 'ffmpeg'}
            self._lastread = None
            self._nframes = float('inf')
            if self._arg_loop and (not self.request._video):
                self._nframes = self.count_frames()
            self._meta['nframes'] = self._nframes
            need_input_fps = need_output_fps = False
            if self.request._video and platform.system().lower() == 'darwin':
                if '-framerate' not in str(self._arg_input_params):
                    need_input_fps = True
                if not self.request.kwargs.get('fps', None):
                    need_output_fps = True
            if need_input_fps:
                self._arg_input_params.extend(['-framerate', str(float(30))])
            if need_output_fps:
                self._arg_output_params.extend(['-r', str(float(30))])
            try:
                self._initialize()
            except IndexError:
                if need_input_fps:
                    self._arg_input_params[-1] = str(float(15))
                    self._initialize()
                else:
                    raise
            if self.request._video:
                self._frame_catcher = FrameCatcher(self._read_gen)

        def _close(self):
            if self._frame_catcher is not None:
                self._frame_catcher.stop_me()
                self._frame_catcher = None
            if self._read_gen is not None:
                self._read_gen.close()
                self._read_gen = None

        def count_frames(self):
            """Count the number of frames. Note that this can take a few
            seconds for large files. Also note that it counts the number
            of frames in the original video and does not take a given fps
            into account.
            """
            cf = self._ffmpeg_api.count_frames_and_secs
            return cf(self._filename)[0]

        def _get_length(self):
            return self._nframes

        def _get_data(self, index):
            """Reads a frame at index. Note for coders: getting an
            arbitrary frame in the video with ffmpeg can be painfully
            slow if some decoding has to be done. This function tries
            to avoid fectching arbitrary frames whenever possible, by
            moving between adjacent frames."""
            if self._arg_loop and self._nframes < float('inf'):
                index %= self._nframes
            if index == self._pos:
                return (self._lastread, dict(new=False))
            elif index < 0:
                raise IndexError('Frame index must be >= 0')
            elif index >= self._nframes:
                raise IndexError('Reached end of video')
            else:
                if index < self._pos or index > self._pos + 100:
                    self._initialize(index)
                else:
                    self._skip_frames(index - self._pos - 1)
                result, is_new = self._read_frame()
                self._pos = index
                return (result, dict(new=is_new))

        def _get_meta_data(self, index):
            return self._meta

        def _initialize(self, index=0):
            if self._read_gen is not None:
                self._read_gen.close()
            iargs = []
            oargs = []
            iargs += self._arg_input_params
            if self.request._video:
                iargs += ['-f', CAM_FORMAT]
                if self._arg_pixelformat:
                    iargs += ['-pix_fmt', self._arg_pixelformat]
                if self._arg_size:
                    iargs += ['-s', self._arg_size]
            elif index > 0:
                starttime = index / self._meta['fps']
                seek_slow = min(10, starttime)
                seek_fast = starttime - seek_slow
                iargs += ['-ss', '%.06f' % seek_fast]
                oargs += ['-ss', '%.06f' % seek_slow]
            if self._arg_size:
                oargs += ['-s', self._arg_size]
            if self.request.kwargs.get('fps', None):
                fps = float(self.request.kwargs['fps'])
                oargs += ['-r', '%.02f' % fps]
            oargs += self._arg_output_params
            pix_fmt = self._pix_fmt
            bpp = self._depth * self._bytes_per_channel
            rf = self._ffmpeg_api.read_frames
            self._read_gen = rf(self._filename, pix_fmt, bpp, input_params=iargs, output_params=oargs)
            if self.request._video:
                try:
                    meta = self._read_gen.__next__()
                except IOError as err:
                    err_text = str(err)
                    if 'darwin' in sys.platform:
                        if "Unknown input format: 'avfoundation'" in err_text:
                            err_text += 'Try installing FFMPEG using home brew to get a version with support for cameras.'
                    raise IndexError('No (working) camera at {}.\n\n{}'.format(self.request._video, err_text))
                else:
                    self._meta.update(meta)
            elif index == 0:
                self._meta.update(self._read_gen.__next__())
            else:
                self._read_gen.__next__()

        def _skip_frames(self, n=1):
            """Reads and throws away n frames"""
            for i in range(n):
                self._read_gen.__next__()
            self._pos += n

        def _read_frame(self):
            w, h = self._meta['size']
            framesize = w * h * self._depth * self._bytes_per_channel
            if self._frame_catcher:
                s, is_new = self._frame_catcher.get_frame()
            else:
                s = self._read_gen.__next__()
                is_new = True
            if len(s) != framesize:
                raise RuntimeError('Frame is %i bytes, but expected %i.' % (len(s), framesize))
            result = np.frombuffer(s, dtype=self._dtype).copy()
            result = result.reshape((h, w, self._depth))
            self._lastread = result
            return (result, is_new)

    class Writer(Format.Writer):
        _write_gen = None

        def _open(self, fps=10, codec='libx264', bitrate=None, pixelformat='yuv420p', ffmpeg_params=None, input_params=None, output_params=None, ffmpeg_log_level='quiet', quality=5, macro_block_size=16, audio_path=None, audio_codec=None):
            self._ffmpeg_api = imageio_ffmpeg
            self._filename = self.request.get_local_filename()
            self._pix_fmt = None
            self._depth = None
            self._size = None

        def _close(self):
            if self._write_gen is not None:
                self._write_gen.close()
                self._write_gen = None

        def _append_data(self, im, meta):
            h, w = im.shape[:2]
            size = (w, h)
            depth = 1 if im.ndim == 2 else im.shape[2]
            im = image_as_uint(im, bitdepth=8)
            if not im.flags.c_contiguous:
                im = np.ascontiguousarray(im)
            if self._size is None:
                map = {1: 'gray', 2: 'gray8a', 3: 'rgb24', 4: 'rgba'}
                self._pix_fmt = map.get(depth, None)
                if self._pix_fmt is None:
                    raise ValueError('Image must have 1, 2, 3 or 4 channels')
                self._size = size
                self._depth = depth
                self._initialize()
            if size != self._size:
                raise ValueError('All images in a movie should have same size')
            if depth != self._depth:
                raise ValueError('All images in a movie should have same number of channels')
            assert self._write_gen is not None
            self._write_gen.send(im)

        def set_meta_data(self, meta):
            raise RuntimeError('The ffmpeg format does not support setting meta data.')

        def _initialize(self):
            if self._write_gen is not None:
                self._write_gen.close()
            fps = self.request.kwargs.get('fps', 10)
            codec = self.request.kwargs.get('codec', None)
            bitrate = self.request.kwargs.get('bitrate', None)
            quality = self.request.kwargs.get('quality', None)
            input_params = self.request.kwargs.get('input_params') or []
            output_params = self.request.kwargs.get('output_params') or []
            output_params += self.request.kwargs.get('ffmpeg_params') or []
            pixelformat = self.request.kwargs.get('pixelformat', None)
            macro_block_size = self.request.kwargs.get('macro_block_size', 16)
            ffmpeg_log_level = self.request.kwargs.get('ffmpeg_log_level', None)
            audio_path = self.request.kwargs.get('audio_path', None)
            audio_codec = self.request.kwargs.get('audio_codec', None)
            macro_block_size = macro_block_size or 1
            self._write_gen = self._ffmpeg_api.write_frames(self._filename, self._size, pix_fmt_in=self._pix_fmt, pix_fmt_out=pixelformat, fps=fps, quality=quality, bitrate=bitrate, codec=codec, macro_block_size=macro_block_size, ffmpeg_log_level=ffmpeg_log_level, input_params=input_params, output_params=output_params, audio_path=audio_path, audio_codec=audio_codec)
            self._write_gen.send(None)