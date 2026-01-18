from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootHarmSafetyCatCategories(_messages.Message):
    """A LearningGenaiRootHarmSafetyCatCategories object.

  Enums:
    CategoriesValueListEntryValuesEnum:

  Fields:
    categories: A CategoriesValueListEntryValuesEnum attribute.
  """

    class CategoriesValueListEntryValuesEnum(_messages.Enum):
        """CategoriesValueListEntryValuesEnum enum type.

    Values:
      SAFETYCAT_CATEGORY_UNSPECIFIED: <no description>
      TOXICITY: SafetyCat categories.
      OBSCENE: <no description>
      SEXUAL: <no description>
      INSULT: <no description>
      IDENTITY_HATE: <no description>
      DEATH_HARM_TRAGEDY: <no description>
      VIOLENCE_ABUSE: <no description>
      FIREARMS_WEAPONS: <no description>
      PUBLIC_SAFETY: <no description>
      HEALTH: <no description>
      RELIGION_BELIEF: <no description>
      DRUGS: <no description>
      WAR_CONFLICT: <no description>
      POLITICS: <no description>
      FINANCE: <no description>
      LEGAL: <no description>
      DANGEROUS: Following categories are only supported in
        SAFETY_CAT_TEXT_V3_PAX model
      DANGEROUS_SEVERITY: <no description>
      HARASSMENT_SEVERITY: <no description>
      HATE_SEVERITY: <no description>
      SEXUAL_SEVERITY: <no description>
    """
        SAFETYCAT_CATEGORY_UNSPECIFIED = 0
        TOXICITY = 1
        OBSCENE = 2
        SEXUAL = 3
        INSULT = 4
        IDENTITY_HATE = 5
        DEATH_HARM_TRAGEDY = 6
        VIOLENCE_ABUSE = 7
        FIREARMS_WEAPONS = 8
        PUBLIC_SAFETY = 9
        HEALTH = 10
        RELIGION_BELIEF = 11
        DRUGS = 12
        WAR_CONFLICT = 13
        POLITICS = 14
        FINANCE = 15
        LEGAL = 16
        DANGEROUS = 17
        DANGEROUS_SEVERITY = 18
        HARASSMENT_SEVERITY = 19
        HATE_SEVERITY = 20
        SEXUAL_SEVERITY = 21
    categories = _messages.EnumField('CategoriesValueListEntryValuesEnum', 1, repeated=True)