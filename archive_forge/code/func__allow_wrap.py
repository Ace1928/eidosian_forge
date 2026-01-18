import re
import math
def _allow_wrap(self):
    if self._should_wrap():
        self.__parent.add_new_line()
        next = self.__parent.current_line
        next.set_indent(self.__wrap_point_indent_count, self.__wrap_point_alignment_count)
        next.__items = self.__items[self.__wrap_point_index:]
        self.__items = self.__items[:self.__wrap_point_index]
        next.__character_count += self.__character_count - self.__wrap_point_character_count
        self.__character_count = self.__wrap_point_character_count
        if next.__items[0] == ' ':
            next.__items.pop(0)
            next.__character_count -= 1
        return True
    return False