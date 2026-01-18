import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
@staticmethod
def __make_char_states(text: str, encoding: Encoding, annotations: AnnotationList) -> List[CharState]:
    """
        For each character in the original text, we emit a tuple representing it's "state":

            * which token_ix it corresponds to
            * which word_ix it corresponds to
            * which annotation_ix it corresponds to

        Args:
            text (:obj:`str`):
                The raw text we want to align to

            annotations (:obj:`List[Annotation]`):
                A (possibly empty) list of annotations

            encoding: (:class:`~tokenizers.Encoding`):
                The encoding returned from the tokenizer

        Returns:
            :obj:`List[CharState]`: A list of CharStates, indicating for each char in the text what
            it's state is
        """
    annotation_map = EncodingVisualizer.__make_anno_map(text, annotations)
    char_states: List[CharState] = [CharState(char_ix) for char_ix in range(len(text))]
    for token_ix, token in enumerate(encoding.tokens):
        offsets = encoding.token_to_chars(token_ix)
        if offsets is not None:
            start, end = offsets
            for i in range(start, end):
                char_states[i].tokens.append(token_ix)
    for char_ix, anno_ix in enumerate(annotation_map):
        char_states[char_ix].anno_ix = anno_ix
    return char_states