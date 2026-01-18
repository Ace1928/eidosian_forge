import errno
import shlex
import subprocess
class FFprobe(FFmpeg):
    """Wrapper for `ffprobe <https://www.ffmpeg.org/ffprobe.html>`_."""

    def __init__(self, executable='ffprobe', global_options='', inputs=None):
        """Create an instance of FFprobe.

        Compiles FFprobe command line from passed arguments (executable path, options, inputs).
        FFprobe executable by default is taken from ``PATH`` but can be overridden with an
        absolute path. For more info about FFprobe command line format see
        `here <https://ffmpeg.org/ffprobe.html#Synopsis>`_.

        :param str executable: absolute path to ffprobe executable
        :param iterable global_options: global options passed to ffmpeg executable; can be specified
            either as a list/tuple of strings or a space-separated string
        :param dict inputs: a dictionary specifying one or more inputs as keys with their
            corresponding options as values
        """
        super(FFprobe, self).__init__(executable=executable, global_options=global_options, inputs=inputs)