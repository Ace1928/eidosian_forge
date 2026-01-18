from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoodValueValuesEnum(_messages.Enum):
    """The grammatical mood.

    Values:
      MOOD_UNKNOWN: Mood is not applicable in the analyzed language or is not
        predicted.
      CONDITIONAL_MOOD: Conditional
      IMPERATIVE: Imperative
      INDICATIVE: Indicative
      INTERROGATIVE: Interrogative
      JUSSIVE: Jussive
      SUBJUNCTIVE: Subjunctive
    """
    MOOD_UNKNOWN = 0
    CONDITIONAL_MOOD = 1
    IMPERATIVE = 2
    INDICATIVE = 3
    INTERROGATIVE = 4
    JUSSIVE = 5
    SUBJUNCTIVE = 6