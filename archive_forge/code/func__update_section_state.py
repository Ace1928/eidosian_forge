from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _update_section_state(line_info, state):
    """Uses line_info to determine the current section of the docstring.

  Updates state and line_info.remaining.

  Args:
    line_info: Information about the current line.
    state: The state of the parser.
  """
    section_updated = False
    google_section_permitted = _google_section_permitted(line_info, state)
    google_section = google_section_permitted and _google_section(line_info)
    if google_section:
        state.section.format = Formats.GOOGLE
        state.section.title = google_section
        line_info.remaining = _get_after_google_header(line_info)
        line_info.remaining_raw = line_info.remaining
        section_updated = True
    rst_section = _rst_section(line_info)
    if rst_section:
        state.section.format = Formats.RST
        state.section.title = rst_section
        line_info.remaining = _get_after_directive(line_info)
        line_info.remaining_raw = line_info.remaining
        section_updated = True
    numpy_section = _numpy_section(line_info)
    if numpy_section:
        state.section.format = Formats.NUMPY
        state.section.title = numpy_section
        line_info.remaining = ''
        line_info.remaining_raw = line_info.remaining
        section_updated = True
    if section_updated:
        state.section.new = True
        state.section.indentation = line_info.indentation
        state.section.line1_indentation = line_info.next.indentation
    else:
        state.section.new = False