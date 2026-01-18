from typing import Dict, List, Optional, Tuple
import collections
from cirq.circuits._box_drawing_character_data import box_draw_character, BoxDrawCharacterSet
Outputs text containing the diagram.

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
        