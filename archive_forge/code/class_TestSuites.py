from __future__ import annotations
import abc
import dataclasses
import datetime
import decimal
from xml.dom import minidom
from xml.etree import ElementTree as ET
@dataclasses.dataclass
class TestSuites:
    """A collection of test suites."""
    name: str | None = None
    suites: list[TestSuite] = dataclasses.field(default_factory=list)

    @property
    def disabled(self) -> int:
        """The number of disabled test cases."""
        return sum((suite.disabled for suite in self.suites))

    @property
    def errors(self) -> int:
        """The number of test cases containing error info."""
        return sum((suite.errors for suite in self.suites))

    @property
    def failures(self) -> int:
        """The number of test cases containing failure info."""
        return sum((suite.failures for suite in self.suites))

    @property
    def tests(self) -> int:
        """The number of test cases."""
        return sum((suite.tests for suite in self.suites))

    @property
    def time(self) -> decimal.Decimal:
        """The total time from all test cases."""
        return decimal.Decimal(sum((suite.time for suite in self.suites)))

    def get_attributes(self) -> dict[str, str]:
        """Return a dictionary of attributes for this instance."""
        return _attributes(disabled=self.disabled, errors=self.errors, failures=self.failures, name=self.name, tests=self.tests, time=self.time)

    def get_xml_element(self) -> ET.Element:
        """Return an XML element representing this instance."""
        element = ET.Element('testsuites', self.get_attributes())
        element.extend([suite.get_xml_element() for suite in self.suites])
        return element

    def to_pretty_xml(self) -> str:
        """Return a pretty formatted XML string representing this instance."""
        return _pretty_xml(self.get_xml_element())