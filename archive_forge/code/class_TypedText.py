from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
class TypedText(object):
    """Text with a semantic type that will be used for styling."""

    def __init__(self, texts, text_type=None):
        """String of text and a corresponding type to use to style that text.

    Args:
     texts: (list[str]), list of strs or TypedText objects
       that should be styled using text_type.
     text_type: (TextTypes), the semantic type of the text that
       will be used to style text.
    """
        self.texts = texts
        self.text_type = text_type

    def __len__(self):
        length = 0
        for text in self.texts:
            length += len(text)
        return length

    def __add__(self, other):
        texts = [self, other]
        return TypedText(texts)

    def __radd__(self, other):
        texts = [other, self]
        return TypedText(texts)