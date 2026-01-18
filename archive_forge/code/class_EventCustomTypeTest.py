import collections
import time
import unittest
import os
import pygame
class EventCustomTypeTest(unittest.TestCase):
    """Those tests are special in that they need the _custom_event counter to
    be reset before and/or after being run."""

    def setUp(self):
        pygame.quit()
        pygame.init()
        pygame.display.init()

    def tearDown(self):
        pygame.quit()

    def test_custom_type(self):
        self.assertEqual(pygame.event.custom_type(), pygame.USEREVENT + 1)
        atype = pygame.event.custom_type()
        atype2 = pygame.event.custom_type()
        self.assertEqual(atype, atype2 - 1)
        ev = pygame.event.Event(atype)
        pygame.event.post(ev)
        queue = pygame.event.get(atype)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].type, atype)

    def test_custom_type__end_boundary(self):
        """Ensure custom_type() raises error when no more custom types.

        The last allowed custom type number should be (pygame.NUMEVENTS - 1).
        """
        last = -1
        start = pygame.event.custom_type() + 1
        for _ in range(start, pygame.NUMEVENTS):
            last = pygame.event.custom_type()
        self.assertEqual(last, pygame.NUMEVENTS - 1)
        with self.assertRaises(pygame.error):
            pygame.event.custom_type()

    def test_custom_type__reset(self):
        """Ensure custom events get 'deregistered' by quit()."""
        before = pygame.event.custom_type()
        self.assertEqual(before, pygame.event.custom_type() - 1)
        pygame.quit()
        pygame.init()
        pygame.display.init()
        self.assertEqual(before, pygame.event.custom_type())