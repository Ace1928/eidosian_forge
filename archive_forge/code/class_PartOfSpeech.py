from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartOfSpeech(_messages.Message):
    """Represents part of speech information for a token. Parts of speech are
  as defined in http://www.lrec-
  conf.org/proceedings/lrec2012/pdf/274_Paper.pdf

  Enums:
    AspectValueValuesEnum: The grammatical aspect.
    CaseValueValuesEnum: The grammatical case.
    FormValueValuesEnum: The grammatical form.
    GenderValueValuesEnum: The grammatical gender.
    MoodValueValuesEnum: The grammatical mood.
    NumberValueValuesEnum: The grammatical number.
    PersonValueValuesEnum: The grammatical person.
    ProperValueValuesEnum: The grammatical properness.
    ReciprocityValueValuesEnum: The grammatical reciprocity.
    TagValueValuesEnum: The part of speech tag.
    TenseValueValuesEnum: The grammatical tense.
    VoiceValueValuesEnum: The grammatical voice.

  Fields:
    aspect: The grammatical aspect.
    case: The grammatical case.
    form: The grammatical form.
    gender: The grammatical gender.
    mood: The grammatical mood.
    number: The grammatical number.
    person: The grammatical person.
    proper: The grammatical properness.
    reciprocity: The grammatical reciprocity.
    tag: The part of speech tag.
    tense: The grammatical tense.
    voice: The grammatical voice.
  """

    class AspectValueValuesEnum(_messages.Enum):
        """The grammatical aspect.

    Values:
      ASPECT_UNKNOWN: Aspect is not applicable in the analyzed language or is
        not predicted.
      PERFECTIVE: Perfective
      IMPERFECTIVE: Imperfective
      PROGRESSIVE: Progressive
    """
        ASPECT_UNKNOWN = 0
        PERFECTIVE = 1
        IMPERFECTIVE = 2
        PROGRESSIVE = 3

    class CaseValueValuesEnum(_messages.Enum):
        """The grammatical case.

    Values:
      CASE_UNKNOWN: Case is not applicable in the analyzed language or is not
        predicted.
      ACCUSATIVE: Accusative
      ADVERBIAL: Adverbial
      COMPLEMENTIVE: Complementive
      DATIVE: Dative
      GENITIVE: Genitive
      INSTRUMENTAL: Instrumental
      LOCATIVE: Locative
      NOMINATIVE: Nominative
      OBLIQUE: Oblique
      PARTITIVE: Partitive
      PREPOSITIONAL: Prepositional
      REFLEXIVE_CASE: Reflexive
      RELATIVE_CASE: Relative
      VOCATIVE: Vocative
    """
        CASE_UNKNOWN = 0
        ACCUSATIVE = 1
        ADVERBIAL = 2
        COMPLEMENTIVE = 3
        DATIVE = 4
        GENITIVE = 5
        INSTRUMENTAL = 6
        LOCATIVE = 7
        NOMINATIVE = 8
        OBLIQUE = 9
        PARTITIVE = 10
        PREPOSITIONAL = 11
        REFLEXIVE_CASE = 12
        RELATIVE_CASE = 13
        VOCATIVE = 14

    class FormValueValuesEnum(_messages.Enum):
        """The grammatical form.

    Values:
      FORM_UNKNOWN: Form is not applicable in the analyzed language or is not
        predicted.
      ADNOMIAL: Adnomial
      AUXILIARY: Auxiliary
      COMPLEMENTIZER: Complementizer
      FINAL_ENDING: Final ending
      GERUND: Gerund
      REALIS: Realis
      IRREALIS: Irrealis
      SHORT: Short form
      LONG: Long form
      ORDER: Order form
      SPECIFIC: Specific form
    """
        FORM_UNKNOWN = 0
        ADNOMIAL = 1
        AUXILIARY = 2
        COMPLEMENTIZER = 3
        FINAL_ENDING = 4
        GERUND = 5
        REALIS = 6
        IRREALIS = 7
        SHORT = 8
        LONG = 9
        ORDER = 10
        SPECIFIC = 11

    class GenderValueValuesEnum(_messages.Enum):
        """The grammatical gender.

    Values:
      GENDER_UNKNOWN: Gender is not applicable in the analyzed language or is
        not predicted.
      FEMININE: Feminine
      MASCULINE: Masculine
      NEUTER: Neuter
    """
        GENDER_UNKNOWN = 0
        FEMININE = 1
        MASCULINE = 2
        NEUTER = 3

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

    class NumberValueValuesEnum(_messages.Enum):
        """The grammatical number.

    Values:
      NUMBER_UNKNOWN: Number is not applicable in the analyzed language or is
        not predicted.
      SINGULAR: Singular
      PLURAL: Plural
      DUAL: Dual
    """
        NUMBER_UNKNOWN = 0
        SINGULAR = 1
        PLURAL = 2
        DUAL = 3

    class PersonValueValuesEnum(_messages.Enum):
        """The grammatical person.

    Values:
      PERSON_UNKNOWN: Person is not applicable in the analyzed language or is
        not predicted.
      FIRST: First
      SECOND: Second
      THIRD: Third
      REFLEXIVE_PERSON: Reflexive
    """
        PERSON_UNKNOWN = 0
        FIRST = 1
        SECOND = 2
        THIRD = 3
        REFLEXIVE_PERSON = 4

    class ProperValueValuesEnum(_messages.Enum):
        """The grammatical properness.

    Values:
      PROPER_UNKNOWN: Proper is not applicable in the analyzed language or is
        not predicted.
      PROPER: Proper
      NOT_PROPER: Not proper
    """
        PROPER_UNKNOWN = 0
        PROPER = 1
        NOT_PROPER = 2

    class ReciprocityValueValuesEnum(_messages.Enum):
        """The grammatical reciprocity.

    Values:
      RECIPROCITY_UNKNOWN: Reciprocity is not applicable in the analyzed
        language or is not predicted.
      RECIPROCAL: Reciprocal
      NON_RECIPROCAL: Non-reciprocal
    """
        RECIPROCITY_UNKNOWN = 0
        RECIPROCAL = 1
        NON_RECIPROCAL = 2

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

    class TenseValueValuesEnum(_messages.Enum):
        """The grammatical tense.

    Values:
      TENSE_UNKNOWN: Tense is not applicable in the analyzed language or is
        not predicted.
      CONDITIONAL_TENSE: Conditional
      FUTURE: Future
      PAST: Past
      PRESENT: Present
      IMPERFECT: Imperfect
      PLUPERFECT: Pluperfect
    """
        TENSE_UNKNOWN = 0
        CONDITIONAL_TENSE = 1
        FUTURE = 2
        PAST = 3
        PRESENT = 4
        IMPERFECT = 5
        PLUPERFECT = 6

    class VoiceValueValuesEnum(_messages.Enum):
        """The grammatical voice.

    Values:
      VOICE_UNKNOWN: Voice is not applicable in the analyzed language or is
        not predicted.
      ACTIVE: Active
      CAUSATIVE: Causative
      PASSIVE: Passive
    """
        VOICE_UNKNOWN = 0
        ACTIVE = 1
        CAUSATIVE = 2
        PASSIVE = 3
    aspect = _messages.EnumField('AspectValueValuesEnum', 1)
    case = _messages.EnumField('CaseValueValuesEnum', 2)
    form = _messages.EnumField('FormValueValuesEnum', 3)
    gender = _messages.EnumField('GenderValueValuesEnum', 4)
    mood = _messages.EnumField('MoodValueValuesEnum', 5)
    number = _messages.EnumField('NumberValueValuesEnum', 6)
    person = _messages.EnumField('PersonValueValuesEnum', 7)
    proper = _messages.EnumField('ProperValueValuesEnum', 8)
    reciprocity = _messages.EnumField('ReciprocityValueValuesEnum', 9)
    tag = _messages.EnumField('TagValueValuesEnum', 10)
    tense = _messages.EnumField('TenseValueValuesEnum', 11)
    voice = _messages.EnumField('VoiceValueValuesEnum', 12)