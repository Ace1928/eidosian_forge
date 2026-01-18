import abc
import base64
import contextlib
from io import BytesIO, TextIOWrapper
import itertools
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib._animation_data import (
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
class MovieWriter(AbstractMovieWriter):
    """
    Base class for writing movies.

    This is a base class for MovieWriter subclasses that write a movie frame
    data to a pipe. You cannot instantiate this class directly.
    See examples for how to use its subclasses.

    Attributes
    ----------
    frame_format : str
        The format used in writing frame data, defaults to 'rgba'.
    fig : `~matplotlib.figure.Figure`
        The figure to capture data from.
        This must be provided by the subclasses.
    """
    supported_formats = ['rgba']

    def __init__(self, fps=5, codec=None, bitrate=None, extra_args=None, metadata=None):
        """
        Parameters
        ----------
        fps : int, default: 5
            Movie frame rate (per second).
        codec : str or None, default: :rc:`animation.codec`
            The codec to use.
        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.
        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie encoder. These
            arguments are passed last to the encoder, just before the filename. The
            default, None, means to use :rc:`animation.[name-of-encoder]_args` for the
            builtin writers.
        metadata : dict[str, str], default: {}
            A dictionary of keys and values for metadata to include in the
            output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.
        """
        if type(self) is MovieWriter:
            raise TypeError('MovieWriter cannot be instantiated directly. Please use one of its subclasses.')
        super().__init__(fps=fps, metadata=metadata, codec=codec, bitrate=bitrate)
        self.frame_format = self.supported_formats[0]
        self.extra_args = extra_args

    def _adjust_frame_size(self):
        if self.codec == 'h264':
            wo, ho = self.fig.get_size_inches()
            w, h = adjusted_figsize(wo, ho, self.dpi, 2)
            if (wo, ho) != (w, h):
                self.fig.set_size_inches(w, h, forward=True)
                _log.info('figure size in inches has been adjusted from %s x %s to %s x %s', wo, ho, w, h)
        else:
            w, h = self.fig.get_size_inches()
        _log.debug('frame size in pixels is %s x %s', *self.frame_size)
        return (w, h)

    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        self._w, self._h = self._adjust_frame_size()
        self._run()

    def _run(self):
        command = self._args()
        _log.info('MovieWriter._run: running command: %s', cbook._pformat_subprocess(command))
        PIPE = subprocess.PIPE
        self._proc = subprocess.Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, creationflags=subprocess_creation_flags)

    def finish(self):
        """Finish any processing for writing the movie."""
        out, err = self._proc.communicate()
        out = TextIOWrapper(BytesIO(out)).read()
        err = TextIOWrapper(BytesIO(err)).read()
        if out:
            _log.log(logging.WARNING if self._proc.returncode else logging.DEBUG, 'MovieWriter stdout:\n%s', out)
        if err:
            _log.log(logging.WARNING if self._proc.returncode else logging.DEBUG, 'MovieWriter stderr:\n%s', err)
        if self._proc.returncode:
            raise subprocess.CalledProcessError(self._proc.returncode, self._proc.args, out, err)

    def grab_frame(self, **savefig_kwargs):
        _validate_grabframe_kwargs(savefig_kwargs)
        _log.debug('MovieWriter.grab_frame: Grabbing frame.')
        self.fig.set_size_inches(self._w, self._h)
        self.fig.savefig(self._proc.stdin, format=self.frame_format, dpi=self.dpi, **savefig_kwargs)

    def _args(self):
        """Assemble list of encoder-specific command-line arguments."""
        return NotImplementedError('args needs to be implemented by subclass.')

    @classmethod
    def bin_path(cls):
        """
        Return the binary path to the commandline tool used by a specific
        subclass. This is a class method so that the tool can be looked for
        before making a particular MovieWriter subclass available.
        """
        return str(mpl.rcParams[cls._exec_key])

    @classmethod
    def isAvailable(cls):
        """Return whether a MovieWriter subclass is actually available."""
        return shutil.which(cls.bin_path()) is not None