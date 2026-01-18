import os
import platform
import unittest
import pygame
import time
def _wait_delay_check(self, func_to_check, millis, iterations, delta):
    """ "
        call func_to_check(millis) "iterations" times and check each time if
        function "waited" for given millisecond (+- delta). At the end, take
        average time for each call (whole_duration/iterations), which should
        be equal to millis (+- delta - acceptable margin of error).
        *Created to avoid code duplication during delay and wait tests
        """
    start_time = time.time()
    for i in range(iterations):
        wait_time = func_to_check(millis)
        self.assertAlmostEqual(wait_time, millis, delta=delta)
    stop_time = time.time()
    duration = round((stop_time - start_time) * 1000)
    self.assertAlmostEqual(duration / iterations, millis, delta=delta)