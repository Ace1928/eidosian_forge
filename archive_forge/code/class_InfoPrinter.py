import re
import textwrap
import param
from param.ipython import ParamPager
from param.parameterized import bothmethod
from .util import group_sanitizer, label_sanitizer
class InfoPrinter:
    """
    Class for printing other information related to an object that is
    of use to the user.
    """
    headings = ['\x1b[1;35m%s\x1b[0m', '\x1b[1;32m%s\x1b[0m']
    ansi_escape = re.compile('\\x1b[^m]*m')
    ppager = ParamPager()
    store = None
    elements = []

    @classmethod
    def get_parameter_info(cls, obj, ansi=False, show_values=True, pattern=None, max_col_len=40):
        """
        Get parameter information from the supplied class or object.
        """
        if cls.ppager is None:
            return ''
        if pattern is not None:
            obj = ParamFilter(obj, ParamFilter.regexp_filter(pattern))
            if len(list(obj.param)) <= 1:
                return None
        param_info = cls.ppager.get_param_info(obj)
        param_list = cls.ppager.param_docstrings(param_info)
        if not show_values:
            retval = cls.ansi_escape.sub('', param_list) if not ansi else param_list
            return cls.highlight(pattern, retval)
        else:
            info = cls.ppager(obj)
            if ansi is False:
                info = cls.ansi_escape.sub('', info)
            return cls.highlight(pattern, info)

    @classmethod
    def heading(cls, heading_text, char='=', level=0, ansi=False):
        """
        Turn the supplied heading text into a suitable heading with
        optional underline and color.
        """
        heading_color = cls.headings[level] if ansi else '%s'
        if char is None:
            return heading_color % f'{heading_text}\n'
        else:
            heading_ul = char * len(heading_text)
            return heading_color % f'{heading_ul}\n{heading_text}\n{heading_ul}'

    @classmethod
    def highlight(cls, pattern, string):
        if pattern is None:
            return string
        return re.sub(pattern, '\x1b[43;1;30m\\g<0>\x1b[0m', string, flags=re.IGNORECASE)

    @classmethod
    def info(cls, obj, ansi=False, backend='matplotlib', visualization=True, pattern=None, elements=None):
        """
        Show information about an object in the given category. ANSI
        color codes may be enabled or disabled.
        """
        if elements is None:
            elements = []
        cls.elements = elements
        ansi_escape = re.compile('\\x1b[^m]*m')
        isclass = isinstance(obj, type)
        name = obj.__name__ if isclass else obj.__class__.__name__
        backend_registry = cls.store.registry.get(backend, {})
        plot_class = backend_registry.get(obj if isclass else type(obj), None)
        if hasattr(plot_class, 'plot_classes'):
            plot_class = next(iter(plot_class.plot_classes.values()))
        if visualization is False or plot_class is None:
            if pattern is not None:
                obj = ParamFilter(obj, ParamFilter.regexp_filter(pattern))
                if len(list(obj.param)) <= 1:
                    return f'No {name!r} parameters found matching specified pattern {pattern!r}'
            info = param.ipython.ParamPager()(obj)
            if ansi is False:
                info = ansi_escape.sub('', info)
            return cls.highlight(pattern, info)
        heading = name if isclass else f'{name}: {obj.group} {obj.label}'
        prefix = heading
        lines = [prefix, cls.object_info(obj, name, backend=backend, ansi=ansi)]
        if not isclass:
            lines += ['', cls.target_info(obj, ansi=ansi)]
        if plot_class is not None:
            lines += ['', cls.options_info(plot_class, ansi, pattern=pattern)]
        return '\n'.join(lines)

    @classmethod
    def get_target(cls, obj):
        objtype = obj.__class__.__name__
        group = group_sanitizer(obj.group)
        label = '.' + label_sanitizer(obj.label) if obj.label else ''
        target = f'{objtype}.{group}{label}'
        return (None, target) if hasattr(obj, 'values') else (target, None)

    @classmethod
    def target_info(cls, obj, ansi=False):
        if isinstance(obj, type):
            return ''
        targets = obj.traverse(cls.get_target)
        elements, containers = zip(*targets)
        element_set = {el for el in elements if el is not None}
        container_set = {c for c in containers if c is not None}
        element_info = None
        if len(element_set) == 1:
            element_info = f'Element: {next(iter(element_set))}'
        elif len(element_set) > 1:
            element_info = 'Elements:\n   %s' % '\n   '.join(sorted(element_set))
        container_info = None
        if len(container_set) == 1:
            container_info = f'Container: {next(iter(container_set))}'
        elif len(container_set) > 1:
            container_info = 'Containers:\n   %s' % '\n   '.join(sorted(container_set))
        heading = cls.heading('Target Specifications', ansi=ansi, char='-')
        target_header = '\nTargets in this object available for customization:\n'
        if element_info and container_info:
            target_info = f'{element_info}\n\n{container_info}'
        else:
            target_info = element_info if element_info else container_info
        target_footer = '\nTo see the options info for one of these target specifications,\nwhich are of the form {type}[.{group}[.{label}]], do holoviews.help({type}).'
        return f'{heading}\n{target_header}\n{target_info}\n{target_footer}'

    @classmethod
    def object_info(cls, obj, name, backend, ansi=False):
        element = not getattr(obj, '_deep_indexable', False)
        element_url = 'http://holoviews.org/reference/elements/{backend}/{obj}.html'
        container_url = 'http://holoviews.org/reference/containers/{backend}/{obj}.html'
        url = element_url if element else container_url
        link = url.format(obj=name, backend=backend)
        link = None if element and name not in cls.elements else link
        msg = '\nOnline example: {link}' if link else '' + '\nHelp for the data object: holoviews.help({obj})' + ' or holoviews.help(<{lower}_instance>)'
        return '\n'.join([msg.format(obj=name, lower=name.lower(), link=link)])

    @classmethod
    def options_info(cls, plot_class, ansi=False, pattern=None):
        if plot_class.style_opts:
            backend_name = plot_class.backend
            style_info = f"\n(Consult {backend_name}'s documentation for more information.)"
            style_keywords = f'\t{', '.join(plot_class.style_opts)}'
            style_msg = f'{style_keywords}\n{style_info}'
        else:
            style_msg = '\t<No style options available>'
        param_info = cls.get_parameter_info(plot_class, ansi=ansi, pattern=pattern)
        lines = [cls.heading('Style Options', ansi=ansi, char='-'), '', style_msg, '', cls.heading('Plot Options', ansi=ansi, char='-'), '']
        if param_info is not None:
            lines += ['The plot options are the parameters of the plotting class:\n', param_info]
        elif pattern is not None:
            lines += [f'No {plot_class.__name__!r} parameters found matching specified pattern {pattern!r}.']
        else:
            lines += [f'No {plot_class.__name__!r} parameters found.']
        return '\n'.join(lines)