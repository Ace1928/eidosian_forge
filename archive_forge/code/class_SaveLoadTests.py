from io import BytesIO, StringIO
from typing import IO, Any, List, Optional, Sequence, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._flatten import extractField
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._json import (
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
class SaveLoadTests(TestCase):
    """
    Tests for loading and saving log events.
    """

    def savedEventJSON(self, event: LogEvent) -> str:
        """
        Serialize some an events, assert some things about it, and return the
        JSON.

        @param event: An event.

        @return: JSON.
        """
        return savedJSONInvariants(self, eventAsJSON(event))

    def test_simpleSaveLoad(self) -> None:
        """
        Saving and loading an empty dictionary results in an empty dictionary.
        """
        self.assertEqual(eventFromJSON(self.savedEventJSON({})), {})

    def test_saveLoad(self) -> None:
        """
        Saving and loading a dictionary with some simple values in it results
        in those same simple values in the output; according to JSON's rules,
        though, all dictionary keys must be L{str} and any non-L{str}
        keys will be converted.
        """
        self.assertEqual(eventFromJSON(self.savedEventJSON({1: 2, '3': '4'})), {'1': 2, '3': '4'})

    def test_saveUnPersistable(self) -> None:
        """
        Saving and loading an object which cannot be represented in JSON will
        result in a placeholder.
        """
        self.assertEqual(eventFromJSON(self.savedEventJSON({'1': 2, '3': object()})), {'1': 2, '3': {'unpersistable': True}})

    def test_saveNonASCII(self) -> None:
        """
        Non-ASCII keys and values can be saved and loaded.
        """
        self.assertEqual(eventFromJSON(self.savedEventJSON({'ሴ': '䌡', '3': object()})), {'ሴ': '䌡', '3': {'unpersistable': True}})

    def test_saveBytes(self) -> None:
        """
        Any L{bytes} objects will be saved as if they are latin-1 so they can
        be faithfully re-loaded.
        """
        inputEvent = {'hello': bytes(range(255))}
        inputEvent.update({b'skipped': 'okay'})
        self.assertEqual(eventFromJSON(self.savedEventJSON(inputEvent)), {'hello': bytes(range(255)).decode('charmap')})

    def test_saveUnPersistableThenFormat(self) -> None:
        """
        Saving and loading an object which cannot be represented in JSON, but
        has a string representation which I{can} be saved as JSON, will result
        in the same string formatting; any extractable fields will retain their
        data types.
        """

        class Reprable:

            def __init__(self, value: object) -> None:
                self.value = value

            def __repr__(self) -> str:
                return 'reprable'
        inputEvent = {'log_format': '{object} {object.value}', 'object': Reprable(7)}
        outputEvent = eventFromJSON(self.savedEventJSON(inputEvent))
        self.assertEqual(formatEvent(outputEvent), 'reprable 7')

    def test_extractingFieldsPostLoad(self) -> None:
        """
        L{extractField} can extract fields from an object that's been saved and
        loaded from JSON.
        """

        class Obj:

            def __init__(self) -> None:
                self.value = 345
        inputEvent = dict(log_format='{object.value}', object=Obj())
        loadedEvent = eventFromJSON(self.savedEventJSON(inputEvent))
        self.assertEqual(extractField('object.value', loadedEvent), 345)
        self.assertRaises(KeyError, extractField, 'object', loadedEvent)
        self.assertRaises(KeyError, extractField, 'object', inputEvent)

    def test_failureStructurePreserved(self) -> None:
        """
        Round-tripping a failure through L{eventAsJSON} preserves its class and
        structure.
        """
        events: List[LogEvent] = []
        log = Logger(observer=cast(ILogObserver, events.append))
        try:
            1 / 0
        except ZeroDivisionError:
            f = Failure()
            log.failure('a message about failure', f)
        self.assertEqual(len(events), 1)
        loaded = eventFromJSON(self.savedEventJSON(events[0]))['log_failure']
        self.assertIsInstance(loaded, Failure)
        self.assertTrue(loaded.check(ZeroDivisionError))
        self.assertIsInstance(loaded.getTraceback(), str)

    def test_saveLoadLevel(self) -> None:
        """
        It's important that the C{log_level} key remain a
        L{constantly.NamedConstant} object.
        """
        inputEvent = dict(log_level=LogLevel.warn)
        loadedEvent = eventFromJSON(self.savedEventJSON(inputEvent))
        self.assertIs(loadedEvent['log_level'], LogLevel.warn)

    def test_saveLoadUnknownLevel(self) -> None:
        """
        If a saved bit of JSON (let's say, from a future version of Twisted)
        were to persist a different log_level, it will resolve as None.
        """
        loadedEvent = eventFromJSON('{"log_level": {"name": "other", "__class_uuid__": "02E59486-F24D-46AD-8224-3ACDF2A5732A"}}')
        self.assertEqual(loadedEvent, dict(log_level=None))