import unittest
import os
import pytest
class AudioGstreamerTestCase(AudioTestCase):

    def make_sound(self, source):
        from kivy.core.audio import audio_gstreamer
        return audio_gstreamer.SoundGstreamer(source)