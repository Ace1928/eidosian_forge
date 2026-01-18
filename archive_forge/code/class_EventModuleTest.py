import collections
import time
import unittest
import os
import pygame
class EventModuleTest(unittest.TestCase):

    def _assertCountEqual(self, *args, **kwargs):
        self.assertCountEqual(*args, **kwargs)

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

    def setUp(self):
        pygame.display.init()
        pygame.event.clear()

    def tearDown(self):
        pygame.event.clear()
        pygame.display.quit()

    def test_event_numevents(self):
        """Ensures NUMEVENTS does not exceed the maximum SDL number of events."""
        MAX_SDL_EVENTS = 65535
        self.assertLessEqual(pygame.NUMEVENTS, MAX_SDL_EVENTS)

    def test_event_attribute(self):
        e1 = pygame.event.Event(pygame.USEREVENT, attr1='attr1')
        self.assertEqual(e1.attr1, 'attr1')

    def test_set_blocked(self):
        """Ensure events can be blocked from the queue."""
        event = EVENT_TYPES[0]
        unblocked_event = EVENT_TYPES[1]
        pygame.event.set_blocked(event)
        self.assertTrue(pygame.event.get_blocked(event))
        self.assertFalse(pygame.event.get_blocked(unblocked_event))
        posted = pygame.event.post(pygame.event.Event(event, **EVENT_TEST_PARAMS[event]))
        self.assertFalse(posted)
        posted = pygame.event.post(pygame.event.Event(unblocked_event, **EVENT_TEST_PARAMS[unblocked_event]))
        self.assertTrue(posted)
        ret = pygame.event.get()
        should_be_blocked = [e for e in ret if e.type == event]
        should_be_allowed_types = [e.type for e in ret if e.type != event]
        self.assertEqual(should_be_blocked, [])
        self.assertTrue(unblocked_event in should_be_allowed_types)

    def test_set_blocked__event_sequence(self):
        """Ensure a sequence of event types can be blocked."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.WINDOWFOCUSLOST, pygame.USEREVENT]
        pygame.event.set_blocked(event_types)
        for etype in event_types:
            self.assertTrue(pygame.event.get_blocked(etype))

    def test_set_blocked_all(self):
        """Ensure all events can be unblocked at once."""
        pygame.event.set_blocked(None)
        for e in EVENT_TYPES:
            self.assertTrue(pygame.event.get_blocked(e))

    def test_post__and_poll(self):
        """Ensure events can be posted to the queue."""
        e1 = pygame.event.Event(pygame.USEREVENT, attr1='attr1')
        pygame.event.post(e1)
        posted_event = pygame.event.poll()
        self.assertEqual(e1.attr1, posted_event.attr1, race_condition_notification)
        for i in range(1, 13):
            pygame.event.post(pygame.event.Event(EVENT_TYPES[i], **EVENT_TEST_PARAMS[EVENT_TYPES[i]]))
            self.assertEqual(pygame.event.poll().type, EVENT_TYPES[i], race_condition_notification)

    def test_post_and_get_keydown(self):
        """Ensure keydown events can be posted to the queue."""
        activemodkeys = pygame.key.get_mods()
        events = [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_y, mod=activemodkeys), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, unicode='g'), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode=None), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m, mod=None, window=None), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e, mod=activemodkeys, unicode='e')]
        for e in events:
            pygame.event.post(e)
            posted_event = pygame.event.poll()
            self.assertEqual(e, posted_event, race_condition_notification)

    def test_post_large_user_event(self):
        pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'a': 'a' * 1024}, test=list(range(100))))
        e = pygame.event.poll()
        self.assertEqual(e.type, pygame.USEREVENT)
        self.assertEqual(e.a, 'a' * 1024)
        self.assertEqual(e.test, list(range(100)))

    def test_post_blocked(self):
        """
        Test blocked events are not posted. Also test whether post()
        returns a boolean correctly
        """
        pygame.event.set_blocked(pygame.USEREVENT)
        self.assertFalse(pygame.event.post(pygame.event.Event(pygame.USEREVENT)))
        self.assertFalse(pygame.event.poll())
        pygame.event.set_allowed(pygame.USEREVENT)
        self.assertTrue(pygame.event.post(pygame.event.Event(pygame.USEREVENT)))
        self.assertEqual(pygame.event.poll(), pygame.event.Event(pygame.USEREVENT))

    def test_get(self):
        """Ensure get() retrieves all the events on the queue."""
        event_cnt = 10
        for _ in range(event_cnt):
            pygame.event.post(pygame.event.Event(pygame.USEREVENT))
        queue = pygame.event.get()
        self.assertEqual(len(queue), event_cnt)
        self.assertTrue(all((e.type == pygame.USEREVENT for e in queue)))

    def test_get_type(self):
        ev = pygame.event.Event(pygame.USEREVENT)
        pygame.event.post(ev)
        queue = pygame.event.get(pygame.USEREVENT)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].type, pygame.USEREVENT)
        TESTEVENTS = 10
        for _ in range(TESTEVENTS):
            pygame.event.post(ev)
        q = pygame.event.get([pygame.USEREVENT])
        self.assertEqual(len(q), TESTEVENTS)
        for event in q:
            self.assertEqual(event, ev)

    def test_get_exclude_throw(self):
        self.assertRaises(pygame.error, pygame.event.get, pygame.KEYDOWN, False, pygame.KEYUP)

    def test_get_exclude(self):
        pygame.event.post(pygame.event.Event(pygame.USEREVENT))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN))
        queue = pygame.event.get(exclude=pygame.KEYDOWN)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].type, pygame.USEREVENT)
        pygame.event.post(pygame.event.Event(pygame.KEYUP))
        pygame.event.post(pygame.event.Event(pygame.USEREVENT))
        queue = pygame.event.get(exclude=(pygame.KEYDOWN, pygame.KEYUP))
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].type, pygame.USEREVENT)
        queue = pygame.event.get()
        self.assertEqual(len(queue), 2)

    def test_get__empty_queue(self):
        """Ensure get() works correctly on an empty queue."""
        expected_events = []
        pygame.event.clear()
        retrieved_events = pygame.event.get()
        self.assertListEqual(retrieved_events, expected_events)
        for event_type in EVENT_TYPES:
            retrieved_events = pygame.event.get(event_type)
            self.assertListEqual(retrieved_events, expected_events)
        retrieved_events = pygame.event.get(EVENT_TYPES)
        self.assertListEqual(retrieved_events, expected_events)

    def test_get__event_sequence(self):
        """Ensure get() can handle a sequence of event types."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]
        other_event_type = pygame.MOUSEBUTTONUP
        expected_events = []
        pygame.event.clear()
        retrieved_events = pygame.event.get(event_types)
        self._assertExpectedEvents(expected=expected_events, got=retrieved_events)
        expected_events = []
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(other_event_type, **EVENT_TEST_PARAMS[other_event_type]))
        retrieved_events = pygame.event.get(event_types)
        self._assertExpectedEvents(expected=expected_events, got=retrieved_events)
        expected_events = [pygame.event.Event(event_types[0], **EVENT_TEST_PARAMS[event_types[0]])]
        pygame.event.clear()
        pygame.event.post(expected_events[0])
        retrieved_events = pygame.event.get(event_types)
        self._assertExpectedEvents(expected=expected_events, got=retrieved_events)
        pygame.event.clear()
        expected_events = []
        for etype in event_types:
            expected_events.append(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
            pygame.event.post(expected_events[-1])
        retrieved_events = pygame.event.get(event_types)
        self._assertExpectedEvents(expected=expected_events, got=retrieved_events)

    def test_get_clears_queue(self):
        """Ensure get() clears the event queue after a call"""
        pygame.event.get()
        self.assertEqual(pygame.event.get(), [])

    def test_clear(self):
        """Ensure clear() removes all the events on the queue."""
        for e in EVENT_TYPES:
            pygame.event.post(pygame.event.Event(e, **EVENT_TEST_PARAMS[e]))
        poll_event = pygame.event.poll()
        self.assertNotEqual(poll_event.type, pygame.NOEVENT)
        pygame.event.clear()
        poll_event = pygame.event.poll()
        self.assertEqual(poll_event.type, pygame.NOEVENT, race_condition_notification)

    def test_clear__empty_queue(self):
        """Ensure clear() works correctly on an empty queue."""
        expected_events = []
        pygame.event.clear()
        pygame.event.clear()
        retrieved_events = pygame.event.get()
        self.assertListEqual(retrieved_events, expected_events)

    def test_clear__event_sequence(self):
        """Ensure a sequence of event types can be cleared from the queue."""
        cleared_event_types = EVENT_TYPES[:5]
        expected_event_types = EVENT_TYPES[5:10]
        expected_events = []
        for etype in cleared_event_types:
            pygame.event.post(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
        for etype in expected_events:
            expected_events.append(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
            pygame.event.post(expected_events[-1])
        pygame.event.clear(cleared_event_types)
        remaining_events = pygame.event.get()
        self._assertCountEqual(remaining_events, expected_events)

    def test_event_name(self):
        """Ensure event_name() returns the correct event name."""
        for expected_name, event in NAMES_AND_EVENTS:
            self.assertEqual(pygame.event.event_name(event), expected_name, f'0x{event:X}')

    def test_event_name__userevent_range(self):
        """Ensures event_name() returns the correct name for user events.

        Tests the full range of user events.
        """
        expected_name = 'UserEvent'
        for event in range(pygame.USEREVENT, pygame.NUMEVENTS):
            self.assertEqual(pygame.event.event_name(event), expected_name, f'0x{event:X}')

    def test_event_name__userevent_boundary(self):
        """Ensures event_name() does not return 'UserEvent' for events
        just outside the user event range.
        """
        unexpected_name = 'UserEvent'
        for event in (pygame.USEREVENT - 1, pygame.NUMEVENTS):
            self.assertNotEqual(pygame.event.event_name(event), unexpected_name, f'0x{event:X}')

    def test_event_name__kwargs(self):
        """Ensure event_name() returns the correct event name when kwargs used."""
        for expected_name, event in NAMES_AND_EVENTS:
            self.assertEqual(pygame.event.event_name(type=event), expected_name, f'0x{event:X}')

    def test_peek(self):
        """Ensure queued events can be peeked at."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]
        for event_type in event_types:
            pygame.event.post(pygame.event.Event(event_type, **EVENT_TEST_PARAMS[event_type]))
        for event_type in event_types:
            self.assertTrue(pygame.event.peek(event_type))
        self.assertTrue(pygame.event.peek(event_types))

    def test_peek__event_sequence(self):
        """Ensure peek() can handle a sequence of event types."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]
        other_event_type = pygame.MOUSEBUTTONUP
        pygame.event.clear()
        peeked = pygame.event.peek(event_types)
        self.assertFalse(peeked)
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(other_event_type, **EVENT_TEST_PARAMS[other_event_type]))
        peeked = pygame.event.peek(event_types)
        self.assertFalse(peeked)
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(event_types[0], **EVENT_TEST_PARAMS[event_types[0]]))
        peeked = pygame.event.peek(event_types)
        self.assertTrue(peeked)
        pygame.event.clear()
        for etype in event_types:
            pygame.event.post(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
        peeked = pygame.event.peek(event_types)
        self.assertTrue(peeked)

    def test_peek__empty_queue(self):
        """Ensure peek() works correctly on an empty queue."""
        pygame.event.clear()
        peeked = pygame.event.peek()
        self.assertFalse(peeked)
        for event_type in EVENT_TYPES:
            peeked = pygame.event.peek(event_type)
            self.assertFalse(peeked)
        peeked = pygame.event.peek(EVENT_TYPES)
        self.assertFalse(peeked)

    def test_set_allowed(self):
        """Ensure a blocked event type can be unblocked/allowed."""
        event = EVENT_TYPES[0]
        pygame.event.set_blocked(event)
        self.assertTrue(pygame.event.get_blocked(event))
        pygame.event.set_allowed(event)
        self.assertFalse(pygame.event.get_blocked(event))

    def test_set_allowed__event_sequence(self):
        """Ensure a sequence of blocked event types can be unblocked/allowed."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP]
        pygame.event.set_blocked(event_types)
        pygame.event.set_allowed(event_types)
        for etype in event_types:
            self.assertFalse(pygame.event.get_blocked(etype))

    def test_set_allowed_all(self):
        """Ensure all events can be unblocked/allowed at once."""
        pygame.event.set_blocked(None)
        for e in EVENT_TYPES:
            self.assertTrue(pygame.event.get_blocked(e))
        pygame.event.set_allowed(None)
        for e in EVENT_TYPES:
            self.assertFalse(pygame.event.get_blocked(e))

    def test_pump(self):
        """Ensure pump() functions properly."""
        pygame.event.pump()

    @unittest.skip('flaky test, and broken on 2.0.18 windows')
    def test_set_grab__and_get_symmetric(self):
        """Ensure event grabbing can be enabled and disabled.

        WARNING: Moving the mouse off the display during this test can cause it
                 to fail.
        """
        surf = pygame.display.set_mode((10, 10))
        pygame.event.set_grab(True)
        self.assertTrue(pygame.event.get_grab())
        pygame.event.set_grab(False)
        self.assertFalse(pygame.event.get_grab())

    def test_get_blocked(self):
        """Ensure an event's blocked state can be retrieved."""
        pygame.event.set_allowed(None)
        for etype in EVENT_TYPES:
            blocked = pygame.event.get_blocked(etype)
            self.assertFalse(blocked)
        pygame.event.set_blocked(None)
        for etype in EVENT_TYPES:
            blocked = pygame.event.get_blocked(etype)
            self.assertTrue(blocked)

    def test_get_blocked__event_sequence(self):
        """Ensure get_blocked() can handle a sequence of event types."""
        event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.WINDOWMINIMIZED, pygame.USEREVENT]
        blocked = pygame.event.get_blocked(event_types)
        self.assertFalse(blocked)
        pygame.event.set_blocked(event_types[2])
        blocked = pygame.event.get_blocked(event_types)
        self.assertTrue(blocked)
        pygame.event.set_blocked(event_types)
        blocked = pygame.event.get_blocked(event_types)
        self.assertTrue(blocked)

    @unittest.skip('flaky test, and broken on 2.0.18 windows')
    def test_get_grab(self):
        """Ensure get_grab() works as expected"""
        surf = pygame.display.set_mode((10, 10))
        for i in range(5):
            pygame.event.set_grab(i % 2)
            self.assertEqual(pygame.event.get_grab(), i % 2)

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'requires the SDL_VIDEODRIVER to be a non dummy value')
    @unittest.skipIf(pygame.get_sdl_version() < (2, 0, 16), 'Needs at least SDL 2.0.16')
    def test_set_keyboard_grab_and_get_keyboard_grab(self):
        """Ensure set_keyboard_grab() and get_keyboard_grab() work as expected"""
        surf = pygame.display.set_mode((10, 10))
        pygame.event.set_keyboard_grab(True)
        self.assertTrue(pygame.event.get_keyboard_grab())
        pygame.event.set_keyboard_grab(False)
        self.assertFalse(pygame.event.get_keyboard_grab())

    def test_poll(self):
        """Ensure poll() works as expected"""
        pygame.event.clear()
        ev = pygame.event.poll()
        self.assertEqual(ev.type, pygame.NOEVENT)
        e1 = pygame.event.Event(pygame.USEREVENT)
        e2 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
        e3 = pygame.event.Event(pygame.KEYUP, key=pygame.K_a)
        pygame.event.post(e1)
        pygame.event.post(e2)
        pygame.event.post(e3)
        self.assertEqual(pygame.event.poll().type, e1.type)
        self.assertEqual(pygame.event.poll().type, e2.type)
        self.assertEqual(pygame.event.poll().type, e3.type)
        self.assertEqual(pygame.event.poll().type, pygame.NOEVENT)