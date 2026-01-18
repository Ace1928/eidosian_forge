import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
class RichTextLines:
    """Rich multi-line text.

  Line-by-line text output, with font attributes (e.g., color) and annotations
  (e.g., indices in a multi-dimensional tensor). Used as the text output of CLI
  commands. Can be rendered on terminal environments such as curses.

  This is not to be confused with Rich Text Format (RTF). This class is for text
  lines only.
  """

    def __init__(self, lines, font_attr_segs=None, annotations=None):
        """Constructor of RichTextLines.

    Args:
      lines: A list of str or a single str, representing text output to
        screen. The latter case is for convenience when the text output is
        single-line.
      font_attr_segs: A map from 0-based row index to a list of 3-tuples.
        It lists segments in each row that have special font attributes, such
        as colors, that are not the default attribute. For example:
        {1: [(0, 3, "red"), (4, 7, "green")], 2: [(10, 20, "yellow")]}

        In each tuple, the 1st element is the start index of the segment. The
        2nd element is the end index, in an "open interval" fashion. The 3rd
        element is an object or a list of objects that represents the font
        attribute. Colors are represented as strings as in the examples above.
      annotations: A map from 0-based row index to any object for annotating
        the row. A typical use example is annotating rows of the output as
        indices in a multi-dimensional tensor. For example, consider the
        following text representation of a 3x2x2 tensor:
          [[[0, 0], [0, 0]],
           [[0, 0], [0, 0]],
           [[0, 0], [0, 0]]]
        The annotation can indicate the indices of the first element shown in
        each row, i.e.,
          {0: [0, 0, 0], 1: [1, 0, 0], 2: [2, 0, 0]}
        This information can make display of tensors on screen clearer and can
        help the user navigate (scroll) to the desired location in a large
        tensor.

    Raises:
      ValueError: If lines is of invalid type.
    """
        if isinstance(lines, list):
            self._lines = lines
        elif isinstance(lines, str):
            self._lines = [lines]
        else:
            raise ValueError('Unexpected type in lines: %s' % type(lines))
        self._font_attr_segs = font_attr_segs
        if not self._font_attr_segs:
            self._font_attr_segs = {}
        self._annotations = annotations
        if not self._annotations:
            self._annotations = {}

    @property
    def lines(self):
        return self._lines

    @property
    def font_attr_segs(self):
        return self._font_attr_segs

    @property
    def annotations(self):
        return self._annotations

    def num_lines(self):
        return len(self._lines)

    def slice(self, begin, end):
        """Slice a RichTextLines object.

    The object itself is not changed. A sliced instance is returned.

    Args:
      begin: (int) Beginning line index (inclusive). Must be >= 0.
      end: (int) Ending line index (exclusive). Must be >= 0.

    Returns:
      (RichTextLines) Sliced output instance of RichTextLines.

    Raises:
      ValueError: If begin or end is negative.
    """
        if begin < 0 or end < 0:
            raise ValueError('Encountered negative index.')
        lines = self.lines[begin:end]
        font_attr_segs = {}
        for key in self.font_attr_segs:
            if key >= begin and key < end:
                font_attr_segs[key - begin] = self.font_attr_segs[key]
        annotations = {}
        for key in self.annotations:
            if not isinstance(key, int):
                annotations[key] = self.annotations[key]
            elif key >= begin and key < end:
                annotations[key - begin] = self.annotations[key]
        return RichTextLines(lines, font_attr_segs=font_attr_segs, annotations=annotations)

    def extend(self, other):
        """Extend this instance of RichTextLines with another instance.

    The extension takes effect on the text lines, the font attribute segments,
    as well as the annotations. The line indices in the font attribute
    segments and the annotations are adjusted to account for the existing
    lines. If there are duplicate, non-line-index fields in the annotations,
    the value from the input argument "other" will override that in this
    instance.

    Args:
      other: (RichTextLines) The other RichTextLines instance to be appended at
        the end of this instance.
    """
        orig_num_lines = self.num_lines()
        self._lines.extend(other.lines)
        for line_index in other.font_attr_segs:
            self._font_attr_segs[orig_num_lines + line_index] = other.font_attr_segs[line_index]
        for key in other.annotations:
            if isinstance(key, int):
                self._annotations[orig_num_lines + key] = other.annotations[key]
            else:
                self._annotations[key] = other.annotations[key]

    def _extend_before(self, other):
        """Add another RichTextLines object to the front.

    Args:
      other: (RichTextLines) The other object to add to the front to this
        object.
    """
        other_num_lines = other.num_lines()
        self._lines = other.lines + self._lines
        new_font_attr_segs = {}
        for line_index in self.font_attr_segs:
            new_font_attr_segs[other_num_lines + line_index] = self.font_attr_segs[line_index]
        new_font_attr_segs.update(other.font_attr_segs)
        self._font_attr_segs = new_font_attr_segs
        new_annotations = {}
        for key in self._annotations:
            if isinstance(key, int):
                new_annotations[other_num_lines + key] = self.annotations[key]
            else:
                new_annotations[key] = other.annotations[key]
        new_annotations.update(other.annotations)
        self._annotations = new_annotations

    def append(self, line, font_attr_segs=None):
        """Append a single line of text.

    Args:
      line: (str) The text to be added to the end.
      font_attr_segs: (list of tuples) Font attribute segments of the appended
        line.
    """
        self._lines.append(line)
        if font_attr_segs:
            self._font_attr_segs[len(self._lines) - 1] = font_attr_segs

    def append_rich_line(self, rich_line):
        self.append(rich_line.text, rich_line.font_attr_segs)

    def prepend(self, line, font_attr_segs=None):
        """Prepend (i.e., add to the front) a single line of text.

    Args:
      line: (str) The text to be added to the front.
      font_attr_segs: (list of tuples) Font attribute segments of the appended
        line.
    """
        other = RichTextLines(line)
        if font_attr_segs:
            other.font_attr_segs[0] = font_attr_segs
        self._extend_before(other)

    def write_to_file(self, file_path):
        """Write the object itself to file, in a plain format.

    The font_attr_segs and annotations are ignored.

    Args:
      file_path: (str) path of the file to write to.
    """
        with gfile.Open(file_path, 'w') as f:
            for line in self._lines:
                f.write(line + '\n')