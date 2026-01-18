from typing import TYPE_CHECKING, List, Tuple, cast, Dict
import matplotlib.textpath
import matplotlib.font_manager
def _fit_horizontal(tdd: 'cirq.TextDiagramDrawer', ref_boxwidth: float, col_padding: float) -> Tuple[List[float], List[float]]:
    """Figure out the horizontal spacing of columns to fit everything in.

    Returns:
        col_starts: a list of where (in pixels) each column starts.
        col_widths: a list of each column's width in pixels
    """
    max_xi = max((xi for xi, _ in tdd.entries.keys()))
    max_xi = max(max_xi, max((cast(int, xi2) for _, _, xi2, _, _ in tdd.horizontal_lines)))
    col_widths = [0.0] * (max_xi + 2)
    for (xi, _), v in tdd.entries.items():
        tw = _get_text_width(v.text)
        if tw > col_widths[xi]:
            col_widths[xi] = max(ref_boxwidth, tw)
    for i in range(len(col_widths)):
        padding = tdd.horizontal_padding.get(i, col_padding)
        col_widths[i] += padding
    col_starts = [0.0]
    for i in range(1, max_xi + 3):
        col_starts.append(col_starts[i - 1] + col_widths[i - 1])
    return (col_starts, col_widths)