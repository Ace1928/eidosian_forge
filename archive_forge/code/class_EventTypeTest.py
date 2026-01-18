import collections
import time
import unittest
import os
import pygame
class EventTypeTest(unittest.TestCase):

    def test_Event(self):
        """Ensure an Event object can be created."""
        e = pygame.event.Event(pygame.USEREVENT, some_attr=1, other_attr='1')
        self.assertEqual(e.some_attr, 1)
        self.assertEqual(e.other_attr, '1')
        self.assertEqual(e.type, pygame.USEREVENT)
        self.assertIs(e.dict, e.__dict__)
        e.some_attr = 12
        self.assertEqual(e.some_attr, 12)
        e.new_attr = 15
        self.assertEqual(e.new_attr, 15)
        self.assertRaises(AttributeError, setattr, e, 'type', 0)
        self.assertRaises(AttributeError, setattr, e, 'dict', None)
        d = dir(e)
        attrs = ('type', 'dict', '__dict__', 'some_attr', 'other_attr', 'new_attr')
        for attr in attrs:
            self.assertIn(attr, d)
        self.assertRaises(ValueError, pygame.event.Event, 10, type=100)

    def test_as_str(self):
        try:
            str(pygame.event.Event(EVENT_TYPES[0], a='í'))
        except UnicodeEncodeError:
            self.fail('Event object raised exception for non-ascii character')

    def test_event_bool(self):
        self.assertFalse(pygame.event.Event(pygame.NOEVENT))
        for event_type in [pygame.MOUSEBUTTONDOWN, pygame.ACTIVEEVENT, pygame.WINDOWLEAVE, pygame.USEREVENT_DROPFILE]:
            self.assertTrue(pygame.event.Event(event_type))

    def test_event_equality(self):
        """Ensure that events can be compared correctly."""
        a = pygame.event.Event(EVENT_TYPES[0], a=1)
        b = pygame.event.Event(EVENT_TYPES[0], a=1)
        c = pygame.event.Event(EVENT_TYPES[1], a=1)
        d = pygame.event.Event(EVENT_TYPES[0], a=2)
        self.assertTrue(a == a)
        self.assertFalse(a != a)
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertTrue(a != c)
        self.assertFalse(a == c)
        self.assertTrue(a != d)
        self.assertFalse(a == d)