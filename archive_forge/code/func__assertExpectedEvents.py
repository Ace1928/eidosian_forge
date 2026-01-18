import collections
import time
import unittest
import os
import pygame
def _assertExpectedEvents(self, expected, got):
    """Find events like expected events, raise on unexpected or missing,
        ignore additional event properties if expected properties are present."""
    items_left = got[:]
    for expected_element in expected:
        for item in items_left:
            for key in expected_element.__dict__:
                if item.__dict__[key] != expected_element.__dict__[key]:
                    break
            else:
                items_left.remove(item)
                break
        else:
            raise AssertionError('Expected ' + str(expected_element) + ' among remaining events ' + str(items_left) + ' out of ' + str(got))
    if len(items_left) > 0:
        raise AssertionError('Unexpected Events: ' + str(items_left))