from typing import Dict, List, Optional, Tuple
import collections
from cirq.circuits._box_drawing_character_data import box_draw_character, BoxDrawCharacterSet
class BlockDiagramDrawer:
    """Aligns text and curve data placed onto an abstract 2d grid of blocks."""

    def __init__(self) -> None:
        self._blocks: Dict[Tuple[int, int], Block] = collections.defaultdict(Block)
        self._min_widths: Dict[int, int] = collections.defaultdict(lambda: 0)
        self._min_heights: Dict[int, int] = collections.defaultdict(lambda: 0)
        _ = self._blocks[0, 0]
        _ = self._min_widths[0]
        _ = self._min_heights[0]

    def mutable_block(self, x: int, y: int) -> Block:
        """Returns the block at (x, y) so it can be edited."""
        if x < 0 or y < 0:
            raise IndexError('x < 0 or y < 0')
        return self._blocks[x, y]

    def set_col_min_width(self, x: int, min_width: int):
        """Sets a minimum width for blocks in the column with coordinate x."""
        if x < 0:
            raise IndexError('x < 0')
        self._min_widths[x] = min_width

    def set_row_min_height(self, y: int, min_height: int):
        """Sets a minimum height for blocks in the row with coordinate y."""
        if y < 0:
            raise IndexError('y < 0')
        self._min_heights[y] = min_height

    def render(self, *, block_span_x: Optional[int]=None, block_span_y: Optional[int]=None, min_block_width: int=0, min_block_height: int=0) -> str:
        """Outputs text containing the diagram.

        Args:
            block_span_x: The width of the diagram in blocks. Set to None to
                default to using the smallest width that would include all
                accessed blocks and columns with a specified minimum width.
            block_span_y: The height of the diagram in blocks. Set to None to
                default to using the smallest height that would include all
                accessed blocks and rows with a specified minimum height.
            min_block_width: A global minimum width for all blocks.
            min_block_height: A global minimum height for all blocks.

        Returns:
            The diagram as a string.
        """
        if block_span_x is None:
            block_span_x = 1 + max(max((x for x, _ in self._blocks.keys())), max(self._min_widths.keys()))
        if block_span_y is None:
            block_span_y = 1 + max(max((y for _, y in self._blocks.keys())), max(self._min_heights.keys()))
        empty = Block()

        def block(x: int, y: int) -> Block:
            return self._blocks.get((x, y), empty)
        widths = {x: max(max((block(x, y).min_width() for y in range(block_span_y))), self._min_widths.get(x, 0), min_block_width) for x in range(block_span_x)}
        heights = {y: max(max((block(x, y).min_height() for x in range(block_span_x))), self._min_heights.get(y, 0), min_block_height) for y in range(block_span_y)}
        block_renders = {(x, y): block(x, y).render(widths[x], heights[y]) for x in range(block_span_x) for y in range(block_span_y)}
        out_lines: List[str] = []
        for y in range(block_span_y):
            for by in range(heights[y]):
                out_line_chunks: List[str] = []
                for x in range(block_span_x):
                    out_line_chunks.extend(block_renders[x, y][by])
                out_lines.append(''.join(out_line_chunks).rstrip())
        return '\n'.join(out_lines)