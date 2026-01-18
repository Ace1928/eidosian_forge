import math
from enum import Enum, unique
from typing import Dict, List, Sequence, Tuple, Union
def _add_cache(self, prediction_tokens: List[str], edit_distance: List[List[Tuple[int, _EditOperations]]]) -> None:
    """Add newly computed rows to cache.

        Since edit distance is only calculated on the hypothesis suffix that was not in cache, the number of rows in
        `edit_distance` matrx may be shorter than hypothesis length. In that case we skip over these initial words.

        Args:
            prediction_tokens: A tokenized predicted sentence.
            edit_distance:
                A matrix of the Levenshtedin edit distance. The element part of the matrix is a tuple of an edit
                operation cost and an edit operation itself.

        """
    if self.cache_size >= _MAX_CACHE_SIZE:
        return
    node = self.cache
    skip_num = len(prediction_tokens) - len(edit_distance)
    for i in range(skip_num):
        node = node[prediction_tokens[i]][0]
    for word, row in zip(prediction_tokens[skip_num:], edit_distance):
        if word not in node:
            node[word] = ({}, tuple(row))
            self.cache_size += 1
        value = node[word]
        node = value[0]