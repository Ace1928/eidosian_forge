import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
@staticmethod
def __make_anno_map(text: str, annotations: AnnotationList) -> PartialIntList:
    """
        Args:
            text (:obj:`str`):
                The raw text we want to align to

            annotations (:obj:`AnnotationList`):
                A (possibly empty) list of annotations

        Returns:
            A list of  length len(text) whose entry at index i is None if there is no annotation on
            charachter i or k, the index of the annotation that covers index i where k is with
            respect to the list of annotations
        """
    annotation_map = [None] * len(text)
    for anno_ix, a in enumerate(annotations):
        for i in range(a.start, a.end):
            annotation_map[i] = anno_ix
    return annotation_map