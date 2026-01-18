from tempfile import NamedTemporaryFile, mkdtemp
from os.path import split, join as pjoin, dirname
import pathlib
from unittest import TestCase, mock
import struct
import wave
from io import BytesIO
import pytest
from IPython.lib import display
from IPython.testing.decorators import skipif_not_numpy
@simulate_numpy_not_installed()
class TestAudioDataWithoutNumpy(TestAudioDataWithNumpy):

    @skipif_not_numpy
    def test_audio_raises_for_nested_list(self):
        stereo_signal = [list(get_test_tone())] * 2
        self.assertRaises(TypeError, lambda: display.Audio(stereo_signal, rate=44100))