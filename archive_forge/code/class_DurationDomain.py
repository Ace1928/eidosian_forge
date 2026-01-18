from datetime import datetime, timedelta
from itertools import islice
from operator import itemgetter
from typing import Any, Dict, List, cast
from docutils import nodes
import sphinx
from sphinx.application import Sphinx
from sphinx.domains import Domain
from sphinx.locale import __
from sphinx.util import logging
class DurationDomain(Domain):
    """A domain for durations of Sphinx processing."""
    name = 'duration'

    @property
    def reading_durations(self) -> Dict[str, timedelta]:
        return self.data.setdefault('reading_durations', {})

    def note_reading_duration(self, duration: timedelta) -> None:
        self.reading_durations[self.env.docname] = duration

    def clear(self) -> None:
        self.reading_durations.clear()

    def clear_doc(self, docname: str) -> None:
        self.reading_durations.pop(docname, None)

    def merge_domaindata(self, docnames: List[str], otherdata: Dict[str, timedelta]) -> None:
        for docname, duration in otherdata.items():
            if docname in docnames:
                self.reading_durations[docname] = duration