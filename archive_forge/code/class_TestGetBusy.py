import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
class TestGetBusy(unittest.TestCase):
    """Test pygame.mixer.get_busy.

    |tags:slow|
    """

    def setUp(self):
        pygame.mixer.init()

    def tearDown(self):
        pygame.mixer.quit()

    def test_no_sound_playing(self):
        """
        Test that get_busy returns False when no sound is playing.
        """
        self.assertFalse(pygame.mixer.get_busy())

    def test_one_sound_playing(self):
        """
        Test that get_busy returns True when one sound is playing.
        """
        sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound.play()
        time.sleep(0.2)
        self.assertTrue(pygame.mixer.get_busy())
        sound.stop()

    def test_multiple_sounds_playing(self):
        """
        Test that get_busy returns True when multiple sounds are playing.
        """
        sound1 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound2 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound1.play()
        sound2.play()
        time.sleep(0.2)
        self.assertTrue(pygame.mixer.get_busy())
        sound1.stop()
        sound2.stop()

    def test_all_sounds_stopped(self):
        """
        Test that get_busy returns False when all sounds are stopped.
        """
        sound1 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound2 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound1.play()
        sound2.play()
        time.sleep(0.2)
        sound1.stop()
        sound2.stop()
        time.sleep(0.2)
        self.assertFalse(pygame.mixer.get_busy())

    def test_all_sounds_stopped_with_fadeout(self):
        """
        Test that get_busy returns False when all sounds are stopped with
        fadeout.
        """
        sound1 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound2 = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound1.play()
        sound2.play()
        time.sleep(0.2)
        sound1.fadeout(100)
        sound2.fadeout(100)
        time.sleep(0.3)
        self.assertFalse(pygame.mixer.get_busy())

    def test_sound_fading_out(self):
        """Tests that get_busy() returns True when a sound is fading out"""
        sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        sound.play(fade_ms=1000)
        time.sleep(1.1)
        self.assertTrue(pygame.mixer.get_busy())
        sound.stop()