import unittest
import os
import pytest
class AudioPygameTestCase(AudioTestCase):

    def make_sound(self, source):
        from kivy.core.audio import audio_pygame
        return audio_pygame.SoundPygame(source)