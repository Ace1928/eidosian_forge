from __future__ import annotations
from typing import Any, Mapping, Optional, Union
from pymongo import common
class Collation:
    """Collation

    :Parameters:
      - `locale`: (string) The locale of the collation. This should be a string
        that identifies an `ICU locale ID` exactly. For example, ``en_US`` is
        valid, but ``en_us`` and ``en-US`` are not. Consult the MongoDB
        documentation for a list of supported locales.
      - `caseLevel`: (optional) If ``True``, turn on case sensitivity if
        `strength` is 1 or 2 (case sensitivity is implied if `strength` is
        greater than 2). Defaults to ``False``.
      - `caseFirst`: (optional) Specify that either uppercase or lowercase
        characters take precedence. Must be one of the following values:

          * :data:`~CollationCaseFirst.UPPER`
          * :data:`~CollationCaseFirst.LOWER`
          * :data:`~CollationCaseFirst.OFF` (the default)

      - `strength`: (optional) Specify the comparison strength. This is also
        known as the ICU comparison level. This must be one of the following
        values:

          * :data:`~CollationStrength.PRIMARY`
          * :data:`~CollationStrength.SECONDARY`
          * :data:`~CollationStrength.TERTIARY` (the default)
          * :data:`~CollationStrength.QUATERNARY`
          * :data:`~CollationStrength.IDENTICAL`

        Each successive level builds upon the previous. For example, a
        `strength` of :data:`~CollationStrength.SECONDARY` differentiates
        characters based both on the unadorned base character and its accents.

      - `numericOrdering`: (optional) If ``True``, order numbers numerically
        instead of in collation order (defaults to ``False``).
      - `alternate`: (optional) Specify whether spaces and punctuation are
        considered base characters. This must be one of the following values:

          * :data:`~CollationAlternate.NON_IGNORABLE` (the default)
          * :data:`~CollationAlternate.SHIFTED`

      - `maxVariable`: (optional) When `alternate` is
        :data:`~CollationAlternate.SHIFTED`, this option specifies what
        characters may be ignored. This must be one of the following values:

          * :data:`~CollationMaxVariable.PUNCT` (the default)
          * :data:`~CollationMaxVariable.SPACE`

      - `normalization`: (optional) If ``True``, normalizes text into Unicode
        NFD. Defaults to ``False``.
      - `backwards`: (optional) If ``True``, accents on characters are
        considered from the back of the word to the front, as it is done in some
        French dictionary ordering traditions. Defaults to ``False``.
      - `kwargs`: (optional) Keyword arguments supplying any additional options
        to be sent with this Collation object.

    .. versionadded: 3.4

    """
    __slots__ = ('__document',)

    def __init__(self, locale: str, caseLevel: Optional[bool]=None, caseFirst: Optional[str]=None, strength: Optional[int]=None, numericOrdering: Optional[bool]=None, alternate: Optional[str]=None, maxVariable: Optional[str]=None, normalization: Optional[bool]=None, backwards: Optional[bool]=None, **kwargs: Any) -> None:
        locale = common.validate_string('locale', locale)
        self.__document: dict[str, Any] = {'locale': locale}
        if caseLevel is not None:
            self.__document['caseLevel'] = common.validate_boolean('caseLevel', caseLevel)
        if caseFirst is not None:
            self.__document['caseFirst'] = common.validate_string('caseFirst', caseFirst)
        if strength is not None:
            self.__document['strength'] = common.validate_integer('strength', strength)
        if numericOrdering is not None:
            self.__document['numericOrdering'] = common.validate_boolean('numericOrdering', numericOrdering)
        if alternate is not None:
            self.__document['alternate'] = common.validate_string('alternate', alternate)
        if maxVariable is not None:
            self.__document['maxVariable'] = common.validate_string('maxVariable', maxVariable)
        if normalization is not None:
            self.__document['normalization'] = common.validate_boolean('normalization', normalization)
        if backwards is not None:
            self.__document['backwards'] = common.validate_boolean('backwards', backwards)
        self.__document.update(kwargs)

    @property
    def document(self) -> dict[str, Any]:
        """The document representation of this collation.

        .. note::
          :class:`Collation` is immutable. Mutating the value of
          :attr:`document` does not mutate this :class:`Collation`.
        """
        return self.__document.copy()

    def __repr__(self) -> str:
        document = self.document
        return 'Collation({})'.format(', '.join((f'{key}={document[key]!r}' for key in document)))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Collation):
            return self.document == other.document
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        return not self == other