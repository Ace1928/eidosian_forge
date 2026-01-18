import copy
from ..core.pattern import Pattern
class TemplatablePattern(Pattern):

    def __init__(self, input_scanner, parent=None):
        Pattern.__init__(self, input_scanner, parent)
        self.__template_pattern = None
        self._disabled = TemplateNames()
        self._excluded = TemplateNames()
        if parent is not None:
            self.__template_pattern = self._input.get_regexp(parent.__template_pattern)
            self._disabled = copy.copy(parent._disabled)
            self._excluded = copy.copy(parent._excluded)
        self.__patterns = TemplatePatterns(input_scanner)

    def _create(self):
        return TemplatablePattern(self._input, self)

    def _update(self):
        self.__set_templated_pattern()

    def read_options(self, options):
        result = self._create()
        for language in ['django', 'erb', 'handlebars', 'php', 'smarty', 'angular']:
            setattr(result._disabled, language, not language in options.templating)
        result._update()
        return result

    def disable(self, language):
        result = self._create()
        setattr(result._disabled, language, True)
        result._update()
        return result

    def exclude(self, language):
        result = self._create()
        setattr(result._excluded, language, True)
        result._update()
        return result

    def read(self):
        result = ''
        if bool(self._match_pattern):
            result = self._input.read(self._starting_pattern)
        else:
            result = self._input.read(self._starting_pattern, self.__template_pattern)
        next = self._read_template()
        while bool(next):
            if self._match_pattern is not None:
                next += self._input.read(self._match_pattern)
            else:
                next += self._input.readUntil(self.__template_pattern)
            result += next
            next = self._read_template()
        if self._until_after:
            result += self._input.readUntilAfter(self._until_after)
        return result

    def __set_templated_pattern(self):
        items = list()
        if not self._disabled.php:
            items.append(self.__patterns.php._starting_pattern.pattern)
        if not self._disabled.handlebars:
            items.append(self.__patterns.handlebars._starting_pattern.pattern)
        if not self._disabled.erb:
            items.append(self.__patterns.erb._starting_pattern.pattern)
        if not self._disabled.django:
            items.append(self.__patterns.django._starting_pattern.pattern)
            items.append(self.__patterns.django_value._starting_pattern.pattern)
            items.append(self.__patterns.django_comment._starting_pattern.pattern)
        if not self._disabled.smarty:
            items.append(self.__patterns.smarty._starting_pattern.pattern)
        if self._until_pattern:
            items.append(self._until_pattern.pattern)
        self.__template_pattern = self._input.get_regexp('(?:' + '|'.join(items) + ')')

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