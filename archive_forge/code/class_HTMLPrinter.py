from .numbers import number_to_scientific_html
from .string import StrPrinter
class HTMLPrinter(StrPrinter):
    printmethod_attr = '_html'
    _default_settings = dict(StrPrinter._default_settings, repr_name='html', Equilibrium_arrow='&harr;', Reaction_arrow='&rarr;', Reaction_param_separator=_html_semicolon, magnitude_fmt=number_to_scientific_html)

    def _print_Substance(self, s, **kwargs):
        return s.html_name or s.name

    def _print_ReactionSystem(self, rsys, **kwargs):
        return super(HTMLPrinter, self)._print_ReactionSystem(rsys, **kwargs).replace('\n', '<br>\n')