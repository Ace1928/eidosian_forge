import copy
from ..core.pattern import Pattern
def _read_template(self):
    resulting_string = ''
    c = self._input.peek()
    if c == '<':
        peek1 = self._input.peek(1)
        if not self._disabled.php and (not self._excluded.php) and (peek1 == '?'):
            resulting_string = resulting_string or self.__patterns.php.read()
        if not self._disabled.erb and (not self._excluded.erb) and (peek1 == '%'):
            resulting_string = resulting_string or self.__patterns.erb.read()
    elif c == '{':
        if not self._disabled.handlebars and (not self._excluded.handlebars):
            resulting_string = resulting_string or self.__patterns.handlebars_comment.read()
            resulting_string = resulting_string or self.__patterns.handlebars_unescaped.read()
            resulting_string = resulting_string or self.__patterns.handlebars.read()
        if not self._disabled.django:
            if not self._excluded.django and (not self._excluded.handlebars):
                resulting_string = resulting_string or self.__patterns.django_value.read()
            if not self._excluded.django:
                resulting_string = resulting_string or self.__patterns.django_comment.read()
                resulting_string = resulting_string or self.__patterns.django.read()
        if not self._disabled.smarty:
            if self._disabled.django and self._disabled.handlebars:
                resulting_string = resulting_string or self.__patterns.smarty_comment.read()
                resulting_string = resulting_string or self.__patterns.smarty_literal.read()
                resulting_string = resulting_string or self.__patterns.smarty.read()
    return resulting_string