import queue
import re
import subprocess
import sys
import threading
import time
from io import DEFAULT_BUFFER_SIZE
from .exceptions import DecodeError
from .base import AudioFile
class FFmpegAudioFile(AudioFile):
    """An audio file decoded by the ffmpeg command-line utility."""

    def __init__(self, filename, block_size=DEFAULT_BUFFER_SIZE):
        windows = sys.platform.startswith('win')
        if windows:
            windows_error_mode_lock.acquire()
            SEM_NOGPFAULTERRORBOX = 2
            import ctypes
            previous_error_mode = ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
            ctypes.windll.kernel32.SetErrorMode(previous_error_mode | SEM_NOGPFAULTERRORBOX)
        try:
            self.proc = popen_multiple(COMMANDS, ['-i', filename, '-f', 's16le', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, creationflags=PROC_FLAGS)
        except OSError:
            raise NotInstalledError()
        finally:
            if windows:
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetErrorMode(previous_error_mode)
                finally:
                    windows_error_mode_lock.release()
        self.stdout_reader = QueueReaderThread(self.proc.stdout, block_size)
        self.stdout_reader.start()
        self._get_info()
        self.stderr_reader = QueueReaderThread(self.proc.stderr)
        self.stderr_reader.start()

    def read_data(self, timeout=10.0):
        """Read blocks of raw PCM data from the file."""
        start_time = time.time()
        while True:
            data = None
            try:
                data = self.stdout_reader.queue.get(timeout=timeout)
                if data:
                    yield data
                else:
                    break
            except queue.Empty:
                end_time = time.time()
                if not data:
                    if end_time - start_time >= timeout:
                        raise ReadTimeoutError('ffmpeg output: {}'.format(b''.join(self.stderr_reader.queue.queue)))
                    else:
                        start_time = end_time
                        continue

    def _get_info(self):
        """Reads the tool's output from its stderr stream, extracts the
        relevant information, and parses it.
        """
        out_parts = []
        while True:
            line = self.proc.stderr.readline()
            if not line:
                raise CommunicationError('stream info not found')
            if isinstance(line, bytes):
                line = line.decode('utf8', 'ignore')
            line = line.strip().lower()
            if 'no such file' in line:
                raise OSError('file not found')
            elif 'invalid data found' in line:
                raise UnsupportedError()
            elif 'duration:' in line:
                out_parts.append(line)
            elif 'audio:' in line:
                out_parts.append(line)
                self._parse_info(''.join(out_parts))
                break

    def _parse_info(self, s):
        """Given relevant data from the ffmpeg output, set audio
        parameter fields on this object.
        """
        match = re.search('(\\d+) hz', s)
        if match:
            self.samplerate = int(match.group(1))
        else:
            self.samplerate = 0
        match = re.search('hz, ([^,]+),', s)
        if match:
            mode = match.group(1)
            if mode == 'stereo':
                self.channels = 2
            else:
                cmatch = re.match('(\\d+)\\.?(\\d)?', mode)
                if cmatch:
                    self.channels = sum(map(int, cmatch.group().split('.')))
                else:
                    self.channels = 1
        else:
            self.channels = 0
        match = re.search('duration: (\\d+):(\\d+):(\\d+).(\\d)', s)
        if match:
            durparts = list(map(int, match.groups()))
            duration = durparts[0] * 60 * 60 + durparts[1] * 60 + durparts[2] + float(durparts[3]) / 10
            self.duration = duration
        else:
            self.duration = 0

    def close(self):
        """Close the ffmpeg process used to perform the decoding."""
        if hasattr(self, 'proc'):
            self.proc.poll()
            if self.proc.returncode is None:
                self.proc.kill()
                self.proc.wait()
            if hasattr(self, 'stderr_reader'):
                self.stderr_reader.join()
            if hasattr(self, 'stdout_reader'):
                self.stdout_reader.join()
            self.proc.stdout.close()
            self.proc.stderr.close()

    def __del__(self):
        self.close()

    def __iter__(self):
        return self.read_data()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False