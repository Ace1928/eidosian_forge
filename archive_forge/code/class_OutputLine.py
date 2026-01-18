import re
import math
class OutputLine:

    def __init__(self, parent):
        self.__parent = parent
        self.__character_count = 0
        self.__indent_count = -1
        self.__alignment_count = 0
        self.__wrap_point_index = 0
        self.__wrap_point_character_count = 0
        self.__wrap_point_indent_count = -1
        self.__wrap_point_alignment_count = 0
        self.__items = []

    def clone_empty(self):
        line = OutputLine(self.__parent)
        line.set_indent(self.__indent_count, self.__alignment_count)
        return line

    def item(self, index):
        return self.__items[index]

    def is_empty(self):
        return len(self.__items) == 0

    def set_indent(self, indent=0, alignment=0):
        if self.is_empty():
            self.__indent_count = indent
            self.__alignment_count = alignment
            self.__character_count = self.__parent.get_indent_size(self.__indent_count, self.__alignment_count)

    def _set_wrap_point(self):
        if self.__parent.wrap_line_length:
            self.__wrap_point_index = len(self.__items)
            self.__wrap_point_character_count = self.__character_count
            self.__wrap_point_indent_count = self.__parent.next_line.__indent_count
            self.__wrap_point_alignment_count = self.__parent.next_line.__alignment_count

    def _should_wrap(self):
        return self.__wrap_point_index and self.__character_count > self.__parent.wrap_line_length and (self.__wrap_point_character_count > self.__parent.next_line.__character_count)

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

    def last(self):
        if not self.is_empty():
            return self.__items[-1]
        return None

    def push(self, item):
        self.__items.append(item)
        last_newline_index = item.rfind('\n')
        if last_newline_index != -1:
            self.__character_count = len(item) - last_newline_index
        else:
            self.__character_count += len(item)

    def pop(self):
        item = None
        if not self.is_empty():
            item = self.__items.pop()
            self.__character_count -= len(item)
        return item

    def _remove_indent(self):
        if self.__indent_count > 0:
            self.__indent_count -= 1
            self.__character_count -= self.__parent.indent_size

    def _remove_wrap_indent(self):
        if self.__wrap_point_indent_count > 0:
            self.__wrap_point_indent_count -= 1

    def trim(self):
        while self.last() == ' ':
            self.__items.pop()
            self.__character_count -= 1

    def toString(self):
        result = ''
        if self.is_empty():
            if self.__parent.indent_empty_lines:
                result = self.__parent.get_indent_string(self.__indent_count)
        else:
            result = self.__parent.get_indent_string(self.__indent_count, self.__alignment_count)
            result += ''.join(self.__items)
        return result