from typing import TYPE_CHECKING, List, Tuple, cast, Dict
import matplotlib.textpath
import matplotlib.font_manager
def _fit_vertical(tdd: 'cirq.TextDiagramDrawer', ref_boxheight: float, row_padding: float) -> Tuple[List[float], List[float], Dict[float, int]]:
    """Return data structures used to turn tdd vertical coordinates into
    well-spaced SVG coordinates.

    The eagle eyed coder may notice that this function is very
    similar to _fit_horizontal. That function was written first
    because horizontal spacing is very important for being able
    to see all the gates but vertical spacing is just for aesthetics.
    It wasn't until this function was written that I (mpharrigan)
    noticed that -- unlike the x-coordinates (which are all integers) --
    y-coordinates come in half-integers. Please use yi_map to convert
    TextDiagramDrawer y-values to y indices which can be used to index
    into row_starts and row_heights.

    See gh-2313 to track this (and other) hacks that could be improved.

    Returns:
        row_starts: A list that maps y indices to the starting y position
            (in SVG px)
        row_heights: A list that maps y indices to the height of each row
            (in SVG px). Y-index `yi` goes from row_starts[yi] to
            row_starts[yi] + row_heights[yi]
        yi_map:
            A mapping from half-integer TextDiagramDrawer coordinates
            to integer y indices. Apply this mapping before indexing into
            the former two return values (ie row_starts and row_heights)
    """
    all_yis = sorted({yi for _, yi in tdd.entries.keys()} | {yi1 for _, yi1, _, _, _ in tdd.vertical_lines} | {yi2 for _, _, yi2, _, _ in tdd.vertical_lines} | {yi for yi, _, _, _, _ in tdd.horizontal_lines})
    yi_map = {yi: i for i, yi in enumerate(all_yis)}
    max_yi = max((yi_map[yi] for yi in all_yis))
    row_heights = [0.0] * (max_yi + 2)
    for (_, yi), _ in tdd.entries.items():
        yi = yi_map[yi]
        row_heights[yi] = max(ref_boxheight, row_heights[yi])
    for yi_float in all_yis:
        row_heights[yi_map[yi_float]] += row_padding
    row_starts = [0.0]
    for i in range(1, max_yi + 3):
        row_starts.append(row_starts[i - 1] + row_heights[i - 1])
    return (row_starts, row_heights, yi_map)