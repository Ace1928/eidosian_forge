import math
from enum import Enum, unique
from typing import Dict, List, Sequence, Tuple, Union
def _levenshtein_edit_distance(self, prediction_tokens: List[str], prediction_start: int, cache: List[List[Tuple[int, _EditOperations]]]) -> Tuple[int, List[List[Tuple[int, _EditOperations]]], Tuple[_EditOperations, ...]]:
    """Dynamic programming algorithm to compute the Levenhstein edit distance.

        Args:
            prediction_tokens: A tokenized predicted sentence.
            prediction_start: An index where a predicted sentence to be considered from.
            cache: A cached Levenshtein edit distance.

        Returns:
            Edit distance between the predicted sentence and the reference sentence

        """
    prediction_len = len(prediction_tokens)
    empty_rows: List[List[Tuple[int, _EditOperations]]] = [list(self._get_empty_row(self.reference_len)) for _ in range(prediction_len - prediction_start)]
    edit_distance: List[List[Tuple[int, _EditOperations]]] = cache + empty_rows
    length_ratio = self.reference_len / prediction_len if prediction_tokens else 1.0
    beam_width = math.ceil(length_ratio / 2 + _BEAM_WIDTH) if length_ratio / 2 > _BEAM_WIDTH else _BEAM_WIDTH
    for i in range(prediction_start + 1, prediction_len + 1):
        pseudo_diag = math.floor(i * length_ratio)
        min_j = max(0, pseudo_diag - beam_width)
        max_j = self.reference_len + 1 if i == prediction_len else min(self.reference_len + 1, pseudo_diag + beam_width)
        for j in range(min_j, max_j):
            if j == 0:
                edit_distance[i][j] = (edit_distance[i - 1][j][0] + self.op_delete, _EditOperations.OP_DELETE)
            else:
                if prediction_tokens[i - 1] == self.reference_tokens[j - 1]:
                    cost_substitute = self.op_nothing
                    operation_substitute = _EditOperations.OP_NOTHING
                else:
                    cost_substitute = self.op_substitute
                    operation_substitute = _EditOperations.OP_SUBSTITUTE
                operations = ((edit_distance[i - 1][j - 1][0] + cost_substitute, operation_substitute), (edit_distance[i - 1][j][0] + self.op_delete, _EditOperations.OP_DELETE), (edit_distance[i][j - 1][0] + self.op_insert, _EditOperations.OP_INSERT))
                for operation_cost, operation_name in operations:
                    if edit_distance[i][j][0] > operation_cost:
                        edit_distance[i][j] = (operation_cost, operation_name)
    trace = self._get_trace(prediction_len, edit_distance)
    return (edit_distance[-1][-1][0], edit_distance[len(cache):], trace)