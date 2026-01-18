from fontTools.feaLib.error import FeatureLibError, IncludedFeaNotFound
from fontTools.feaLib.location import FeatureLibLocation
import re
import os
class NonIncludingLexer(IncludingLexer):
    """Lexer that does not follow `include` statements, emits them as-is."""

    def __next__(self):
        return next(self.lexers_[0])