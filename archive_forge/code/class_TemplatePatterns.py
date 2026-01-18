import copy
from ..core.pattern import Pattern
class TemplatePatterns:

    def __init__(self, input_scanner):
        pattern = Pattern(input_scanner)
        self.handlebars_comment = pattern.starting_with('{{!--').until_after('--}}')
        self.handlebars_unescaped = pattern.starting_with('{{{').until_after('}}}')
        self.handlebars = pattern.starting_with('{{').until_after('}}')
        self.php = pattern.starting_with('<\\?(?:[= ]|php)').until_after('\\?>')
        self.erb = pattern.starting_with('<%[^%]').until_after('[^%]%>')
        self.django = pattern.starting_with('{%').until_after('%}')
        self.django_value = pattern.starting_with('{{').until_after('}}')
        self.django_comment = pattern.starting_with('{#').until_after('#}')
        self.smarty_value = pattern.starting_with('{(?=[^}{\\s\\n])').until_after('}')
        self.smarty_comment = pattern.starting_with('{\\*').until_after('\\*}')
        self.smarty_literal = pattern.starting_with('{literal}').until_after('{/literal}')