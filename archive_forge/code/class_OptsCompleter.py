import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
class OptsCompleter:
    """
    Implements the TAB-completion for the %%opts magic.
    """
    _completions = {}

    @classmethod
    def setup_completer(cls):
        """Get the dictionary of valid completions"""
        try:
            for element in Store.options().keys():
                options = Store.options()['.'.join(element)]
                plotkws = options['plot'].allowed_keywords
                stylekws = options['style'].allowed_keywords
                dotted = '.'.join(element)
                cls._completions[dotted] = (plotkws, stylekws if stylekws else [])
        except KeyError:
            pass
        return cls._completions

    @classmethod
    def dotted_completion(cls, line, sorted_keys, compositor_defs):
        """
        Supply the appropriate key in Store.options and supply
        suggestions for further completion.
        """
        completion_key, suggestions = (None, [])
        tokens = [t for t in reversed(line.replace('.', ' ').split())]
        for i, token in enumerate(tokens):
            key_checks = []
            if i >= 0:
                key_checks.append(tokens[i])
            if i >= 1:
                key_checks.append('.'.join([key_checks[-1], tokens[i - 1]]))
            if i >= 2:
                key_checks.append('.'.join([key_checks[-1], tokens[i - 2]]))
            for key in reversed(key_checks):
                if key in sorted_keys:
                    completion_key = key
                    depth = completion_key.count('.')
                    suggestions = [k.split('.')[depth + 1] for k in sorted_keys if k.startswith(completion_key + '.')]
                    return (completion_key, suggestions)
            if token in compositor_defs:
                completion_key = compositor_defs[token]
                break
        return (completion_key, suggestions)

    @classmethod
    def _inside_delims(cls, line, opener, closer):
        return (line.count(opener) - line.count(closer)) % 2

    @classmethod
    def option_completer(cls, k, v):
        """Tab completion hook for the %%opts cell magic."""
        line = v.text_until_cursor
        completions = cls.setup_completer()
        compositor_defs = {el.group: el.output_type.__name__ for el in Compositor.definitions if el.group}
        return cls.line_completer(line, completions, compositor_defs)

    @classmethod
    def line_completer(cls, line, completions, compositor_defs):
        sorted_keys = sorted(completions.keys())
        type_keys = [key for key in sorted_keys if '.' not in key]
        completion_key, suggestions = cls.dotted_completion(line, sorted_keys, compositor_defs)
        verbose_openers = ['style(', 'plot[', 'norm{']
        if suggestions and line.endswith('.'):
            return [f'{completion_key}.{el}' for el in suggestions]
        elif not completion_key:
            return type_keys + list(compositor_defs.keys()) + verbose_openers
        if cls._inside_delims(line, '[', ']'):
            return [kw + '=' for kw in completions[completion_key][0]]
        if cls._inside_delims(line, '{', '}'):
            return ['+axiswise', '+framewise']
        style_completions = [kw + '=' for kw in completions[completion_key][1]]
        if cls._inside_delims(line, '(', ')'):
            return style_completions
        return type_keys + list(compositor_defs.keys()) + verbose_openers