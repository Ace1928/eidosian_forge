import re
import math
def __add_column(self):
    column = len(self.__cache)
    indent = 0
    result = ''
    if self.__indent_size and column >= self.__indent_size:
        indent = int(math.floor(column / self.__indent_size))
        column -= indent * self.__indent_size
        result = indent * self.__indent_string
    if column:
        result += column * ' '
    self.__cache.append(result)