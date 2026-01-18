import errno
import shlex
import subprocess
class FFmpeg(object):
    """Wrapper for various `FFmpeg <https://www.ffmpeg.org/>`_ related applications (ffmpeg,
    ffprobe).
    """

    def __init__(self, executable='ffmpeg', global_options=None, inputs=None, outputs=None):
        """Initialize FFmpeg command line wrapper.

        Compiles FFmpeg command line from passed arguments (executable path, options, inputs and
        outputs). ``inputs`` and ``outputs`` are dictionares containing inputs/outputs as keys and
        their respective options as values. One dictionary value (set of options) must be either a
        single space separated string, or a list or strings without spaces (i.e. each part of the
        option is a separate item of the list, the result of calling ``split()`` on the options
        string). If the value is a list, it cannot be mixed, i.e. cannot contain items with spaces.
        An exception are complex FFmpeg command lines that contain quotes: the quoted part must be
        one string, even if it contains spaces (see *Examples* for more info).
        For more info about FFmpeg command line format see `here
        <https://ffmpeg.org/ffmpeg.html#Synopsis>`_.

        :param str executable: path to ffmpeg executable; by default the ``ffmpeg`` command will be
            searched for in the ``PATH``, but can be overridden with an absolute path to ``ffmpeg``
            executable
        :param iterable global_options: global options passed to ``ffmpeg`` executable (e.g.
            ``-y``, ``-v`` etc.); can be specified either as a list/tuple/set of strings, or one
            space-separated string; by default no global options are passed
        :param dict inputs: a dictionary specifying one or more input arguments as keys with their
            corresponding options (either as a list of strings or a single space separated string) as
            values
        :param dict outputs: a dictionary specifying one or more output arguments as keys with their
            corresponding options (either as a list of strings or a single space separated string) as
            values
        """
        self.executable = executable
        self._cmd = [executable]
        global_options = global_options or []
        if _is_sequence(global_options):
            normalized_global_options = []
            for opt in global_options:
                normalized_global_options += shlex.split(opt)
        else:
            normalized_global_options = shlex.split(global_options)
        self._cmd += normalized_global_options
        self._cmd += _merge_args_opts(inputs, add_input_option=True)
        self._cmd += _merge_args_opts(outputs)
        self.cmd = subprocess.list2cmdline(self._cmd)
        self.process = None

    def __repr__(self):
        return '<{0!r} {1!r}>'.format(self.__class__.__name__, self.cmd)

    def run(self, input_data=None, stdout=None, stderr=None, env=None, **kwargs):
        """Execute FFmpeg command line.

        ``input_data`` can contain input for FFmpeg in case ``pipe`` protocol is used for input.
        ``stdout`` and ``stderr`` specify where to redirect the ``stdout`` and ``stderr`` of the
        process. By default no redirection is done, which means all output goes to running shell
        (this mode should normally only be used for debugging purposes). If FFmpeg ``pipe`` protocol
        is used for output, ``stdout`` must be redirected to a pipe by passing `subprocess.PIPE` as
        ``stdout`` argument. You can pass custom environment to ffmpeg process with ``env``.

        Returns a 2-tuple containing ``stdout`` and ``stderr`` of the process. If there was no
        redirection or if the output was redirected to e.g. `os.devnull`, the value returned will
        be a tuple of two `None` values, otherwise it will contain the actual ``stdout`` and
        ``stderr`` data returned by ffmpeg process.

        More info about ``pipe`` protocol `here <https://ffmpeg.org/ffmpeg-protocols.html#pipe>`_.

        :param str input_data: input data for FFmpeg to deal with (audio, video etc.) as bytes (e.g.
            the result of reading a file in binary mode)
        :param stdout: redirect FFmpeg ``stdout`` there (default is `None` which means no
            redirection)
        :param stderr: redirect FFmpeg ``stderr`` there (default is `None` which means no
            redirection)
        :param env: custom environment for ffmpeg process
        :param kwargs: any other keyword arguments to be forwarded to `subprocess.Popen
            <https://docs.python.org/3/library/subprocess.html#subprocess.Popen>`_
        :return: a 2-tuple containing ``stdout`` and ``stderr`` of the process
        :rtype: tuple
        :raise: `FFRuntimeError` in case FFmpeg command exits with a non-zero code;
            `FFExecutableNotFoundError` in case the executable path passed was not valid
        """
        try:
            self.process = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, env=env, **kwargs)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise FFExecutableNotFoundError("Executable '{0}' not found".format(self.executable))
            else:
                raise
        out = self.process.communicate(input=input_data)
        if self.process.returncode != 0:
            raise FFRuntimeError(self.cmd, self.process.returncode, out[0], out[1])
        return out