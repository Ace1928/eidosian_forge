import re
import math
def _set_wrap_point(self):
    if self.__parent.wrap_line_length:
        self.__wrap_point_index = len(self.__items)
        self.__wrap_point_character_count = self.__character_count
        self.__wrap_point_indent_count = self.__parent.next_line.__indent_count
        self.__wrap_point_alignment_count = self.__parent.next_line.__alignment_count