from pyparsing import *
import_stmt: import_name | import_from
import_name: 'import' dotted_as_names
import_from: 'from' dotted_name 'import' ('*' | '(' import_as_names ')' | import_as_names)
import_as_name: NAME [NAME NAME]
import_as_names: import_as_name (',' import_as_name)* [',']
class SemanticGroup(object):

    def __init__(self, contents):
        self.contents = contents
        while self.contents[-1].__class__ == self.__class__:
            self.contents = self.contents[:-1] + self.contents[-1].contents

    def __str__(self):
        return '{0}({1})'.format(self.label, ' '.join([isinstance(c, str) and c or str(c) for c in self.contents]))