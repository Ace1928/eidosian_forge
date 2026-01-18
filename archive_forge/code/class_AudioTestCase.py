import unittest
import os
import pytest
class AudioTestCase(unittest.TestCase):

    def get_sound(self):
        import os
        assert os.path.exists(SAMPLE_FILE)
        from kivy.core import audio
        return audio.SoundLoader.load(SAMPLE_FILE)

    def test_length_simple(self):
        sound = self.get_sound()
        volume = sound.volume = 0.75
        length = sound.length
        self.assertAlmostEqual(SAMPLE_LENGTH, length, delta=DELTA)
        assert volume == sound.volume

    def test_length_playing(self):
        import time
        sound = self.get_sound()
        sound.play()
        try:
            time.sleep(DELAY)
            length = sound.length
            self.assertAlmostEqual(SAMPLE_LENGTH, length, delta=DELTA)
        finally:
            sound.stop()
        self.assertAlmostEqual(SAMPLE_LENGTH, length, delta=DELTA)

    def test_length_stopped(self):
        import time
        sound = self.get_sound()
        sound.play()
        try:
            time.sleep(DELAY)
        finally:
            sound.stop()
        length = sound.length
        self.assertAlmostEqual(SAMPLE_LENGTH, length, delta=DELTA)