from six import PY2
from functools import wraps
from datetime import datetime, timedelta, tzinfo
def _fromutc(self, dt):
    """
        Given a timezone-aware datetime in a given timezone, calculates a
        timezone-aware datetime in a new timezone.

        Since this is the one time that we *know* we have an unambiguous
        datetime object, we take this opportunity to determine whether the
        datetime is ambiguous and in a "fold" state (e.g. if it's the first
        occurrence, chronologically, of the ambiguous datetime).

        :param dt:
            A timezone-aware :class:`datetime.datetime` object.
        """
    dtoff = dt.utcoffset()
    if dtoff is None:
        raise ValueError('fromutc() requires a non-None utcoffset() result')
    dtdst = dt.dst()
    if dtdst is None:
        raise ValueError('fromutc() requires a non-None dst() result')
    delta = dtoff - dtdst
    dt += delta
    dtdst = enfold(dt, fold=1).dst()
    if dtdst is None:
        raise ValueError('fromutc(): dt.dst gave inconsistent results; cannot convert')
    return dt + dtdst