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
class FileMovieWriter(MovieWriter):
    """
    `MovieWriter` for writing to individual files and stitching at the end.

    This must be sub-classed to be useful.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_format = mpl.rcParams['animation.frame_format']

    def setup(self, fig, outfile, dpi=None, frame_prefix=None):
        """
        Setup for writing the movie file.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure to grab the rendered frames from.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, default: ``fig.dpi``
            The dpi of the output file. This, with the figure size,
            controls the size in pixels of the resulting movie file.
        frame_prefix : str, optional
            The filename prefix to use for temporary files.  If *None* (the
            default), files are written to a temporary directory which is
            deleted by `finish`; if not *None*, no temporary files are
            deleted.
        """
        Path(outfile).parent.resolve(strict=True)
        self.fig = fig
        self.outfile = outfile
        if dpi is None:
            dpi = self.fig.dpi
        self.dpi = dpi
        self._adjust_frame_size()
        if frame_prefix is None:
            self._tmpdir = TemporaryDirectory()
            self.temp_prefix = str(Path(self._tmpdir.name, 'tmp'))
        else:
            self._tmpdir = None
            self.temp_prefix = frame_prefix
        self._frame_counter = 0
        self._temp_paths = list()
        self.fname_format_str = '%s%%07d.%s'

    def __del__(self):
        if hasattr(self, '_tmpdir') and self._tmpdir:
            self._tmpdir.cleanup()

    @property
    def frame_format(self):
        """
        Format (png, jpeg, etc.) to use for saving the frames, which can be
        decided by the individual subclasses.
        """
        return self._frame_format

    @frame_format.setter
    def frame_format(self, frame_format):
        if frame_format in self.supported_formats:
            self._frame_format = frame_format
        else:
            _api.warn_external(f'Ignoring file format {frame_format!r} which is not supported by {type(self).__name__}; using {self.supported_formats[0]} instead.')
            self._frame_format = self.supported_formats[0]

    def _base_temp_name(self):
        return self.fname_format_str % (self.temp_prefix, self.frame_format)

    def grab_frame(self, **savefig_kwargs):
        _validate_grabframe_kwargs(savefig_kwargs)
        path = Path(self._base_temp_name() % self._frame_counter)
        self._temp_paths.append(path)
        self._frame_counter += 1
        _log.debug('FileMovieWriter.grab_frame: Grabbing frame %d to path=%s', self._frame_counter, path)
        with open(path, 'wb') as sink:
            self.fig.savefig(sink, format=self.frame_format, dpi=self.dpi, **savefig_kwargs)

    def finish(self):
        try:
            self._run()
            super().finish()
        finally:
            if self._tmpdir:
                _log.debug('MovieWriter: clearing temporary path=%s', self._tmpdir)
                self._tmpdir.cleanup()