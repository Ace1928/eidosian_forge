import abc
import re
class CamelCaseConvention(Convention):

    def __init__(self):
        self.regex = re.compile('(?!^)_(\\w)', flags=re.UNICODE)

    def convert_function_name(self, name):
        return self._to_camel_case(name)

    def convert_parameter_name(self, name):
        return self._to_camel_case(name)

    def _to_camel_case(self, name):
        return self.regex.sub(lambda m: m.group(1).upper(), name)