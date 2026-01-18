from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TagValueValuesEnum(_messages.Enum):
    """The part of speech tag.

    Values:
      UNKNOWN: Unknown
      ADJ: Adjective
      ADP: Adposition (preposition and postposition)
      ADV: Adverb
      CONJ: Conjunction
      DET: Determiner
      NOUN: Noun (common and proper)
      NUM: Cardinal number
      PRON: Pronoun
      PRT: Particle or other function word
      PUNCT: Punctuation
      VERB: Verb (all tenses and modes)
      X: Other: foreign words, typos, abbreviations
      AFFIX: Affix
    """
    UNKNOWN = 0
    ADJ = 1
    ADP = 2
    ADV = 3
    CONJ = 4
    DET = 5
    NOUN = 6
    NUM = 7
    PRON = 8
    PRT = 9
    PUNCT = 10
    VERB = 11
    X = 12
    AFFIX = 13