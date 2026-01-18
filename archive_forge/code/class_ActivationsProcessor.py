from __future__ import absolute_import, division, print_function
import numpy as np
from ..processors import Processor
from . import beats, chords, downbeats, key, notes, onsets, tempo
from .beats import (BeatDetectionProcessor, BeatTrackingProcessor,
from .chords import (CNNChordFeatureProcessor, CRFChordRecognitionProcessor,
from .downbeats import (RNNDownBeatProcessor, DBNDownBeatTrackingProcessor,
from .key import CNNKeyRecognitionProcessor
from .notes import RNNPianoNoteProcessor, NotePeakPickingProcessor
from .onsets import (CNNOnsetProcessor, OnsetPeakPickingProcessor,
from .tempo import TempoEstimationProcessor
class ActivationsProcessor(Processor):
    """
    ActivationsProcessor processes a file and returns an Activations instance.

    Parameters
    ----------
    mode : {'r', 'w', 'in', 'out', 'load', 'save'}
        Mode of the Processor: read/write.
    fps : float, optional
        Frame rate of the activations (if set, it overwrites the saved frame
        rate).
    sep : str, optional
        Separator between activation values if saved as text file.

    Notes
    -----
    An undefined or empty (“”) separator means that the file should be treated
    as a numpy binary file. Only binary files can store the frame rate of the
    activations.

    """

    def __init__(self, mode, fps=None, sep=None, **kwargs):
        self.mode = mode
        self.fps = fps
        self.sep = sep

    def process(self, data, output=None, **kwargs):
        """
        Depending on the mode, either loads the data stored in the given file
        and returns it as an Activations instance or save the data to the given
        output.

        Parameters
        ----------
        data : str, file handle or numpy array
            Data or file to be loaded (if `mode` is 'r') or data to be saved
            to file (if `mode` is 'w').
        output : str or file handle, optional
            output file (only in write-mode)

        Returns
        -------
        :class:`Activations` instance
            :class:`Activations` instance (only in read-mode)

        """
        if self.mode in ('r', 'in', 'load'):
            return Activations.load(data, fps=self.fps, sep=self.sep)
        if self.mode in ('w', 'out', 'save'):
            Activations(data, fps=self.fps).save(output, sep=self.sep)
        else:
            raise ValueError("wrong mode %s; choose {'r', 'w', 'in', 'out', 'load', 'save'}")
        return data

    @staticmethod
    def add_arguments(parser):
        """
        Add options to save/load activations to an existing parser.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser.

        Returns
        -------
        parser_group : argparse argument group
            Input/output argument parser group.

        """
        g = parser.add_argument_group('save/load the activations')
        g.add_argument('--save', action='store_true', default=False, help='save the activations to file')
        g.add_argument('--load', action='store_true', default=False, help='load the activations from file')
        g.add_argument('--sep', action='store', default=None, help='separator for saving/loading the activations [default: None, i.e. numpy binary format]')
        return g