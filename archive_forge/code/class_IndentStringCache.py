import re
import math
class IndentStringCache:

    def __init__(self, options, base_string):
        self.__cache = ['']
        self.__indent_size = options.indent_size
        self.__indent_string = options.indent_char
        if not options.indent_with_tabs:
            self.__indent_string = options.indent_char * options.indent_size
        base_string = base_string or ''
        if options.indent_level > 0:
            base_string = options.indent_level * self.__indent_string
        self.__base_string = base_string
        self.__base_string_length = len(base_string)

    def get_indent_size(self, indent, column=0):
        result = self.__base_string_length
        if indent < 0:
            result = 0
        result += indent * self.__indent_size
        result += column
        return result

    def get_indent_string(self, indent_level, column=0):
        result = self.__base_string
        if indent_level < 0:
            indent_level = 0
            result = ''
        column += indent_level * self.__indent_size
        self.__ensure_cache(column)
        result += self.__cache[column]
        return result

    def __ensure_cache(self, column):
        while column >= len(self.__cache):
            self.__add_column()

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