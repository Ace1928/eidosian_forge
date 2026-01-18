from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubstitutionOptionValueValuesEnum(_messages.Enum):
    """Option to specify behavior when there is an error in the substitution
    checks. NOTE: this is always set to ALLOW_LOOSE for triggered builds and
    cannot be overridden in the build configuration file.

    Values:
      MUST_MATCH: Fails the build if error in substitutions checks, like
        missing a substitution in the template or in the map.
      ALLOW_LOOSE: Do not fail the build if error in substitutions checks.
    """
    MUST_MATCH = 0
    ALLOW_LOOSE = 1