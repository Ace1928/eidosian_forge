import re
import math
def _remove_indent(self):
    if self.__indent_count > 0:
        self.__indent_count -= 1
        self.__character_count -= self.__parent.indent_size