import os
import platform
import unittest
import pygame
import time
def _fps_test(self, clock, fps, delta):
    """ticks fps times each second, hence get_fps() should return fps"""
    delay_per_frame = 1.0 / fps
    for f in range(fps):
        clock.tick()
        time.sleep(delay_per_frame)
    self.assertAlmostEqual(clock.get_fps(), fps, delta=fps * delta)