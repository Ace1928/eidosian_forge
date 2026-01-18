import re
import textwrap
import wrapt
from deprecated.classic import ClassicAdapter
from deprecated.classic import deprecated as _classic_deprecated
class SphinxAdapter(ClassicAdapter):
    """
    Sphinx adapter -- *for advanced usage only*

    This adapter override the :class:`~deprecated.classic.ClassicAdapter`
    in order to add the Sphinx directives to the end of the function/class docstring.
    Such a directive is a `Paragraph-level markup <http://www.sphinx-doc.org/en/stable/markup/para.html>`_

    - The directive can be one of "versionadded", "versionchanged" or "deprecated".
    - The version number is added if provided.
    - The reason message is obviously added in the directive block if not empty.
    """

    def __init__(self, directive, reason='', version='', action=None, category=DeprecationWarning, line_length=70):
        """
        Construct a wrapper adapter.

        :type  directive: str
        :param directive:
            Sphinx directive: can be one of "versionadded", "versionchanged" or "deprecated".

        :type  reason: str
        :param reason:
            Reason message which documents the deprecation in your library (can be omitted).

        :type  version: str
        :param version:
            Version of your project which deprecates this feature.
            If you follow the `Semantic Versioning <https://semver.org/>`_,
            the version number has the format "MAJOR.MINOR.PATCH".

        :type  action: str
        :param action:
            A warning filter used to activate or not the deprecation warning.
            Can be one of "error", "ignore", "always", "default", "module", or "once".
            If ``None`` or empty, the the global filtering mechanism is used.
            See: `The Warnings Filter`_ in the Python documentation.

        :type  category: type
        :param category:
            The warning category to use for the deprecation warning.
            By default, the category class is :class:`~DeprecationWarning`,
            you can inherit this class to define your own deprecation warning category.

        :type  line_length: int
        :param line_length:
            Max line length of the directive text. If non nul, a long text is wrapped in several lines.
        """
        if not version:
            raise ValueError("'version' argument is required in Sphinx directives")
        self.directive = directive
        self.line_length = line_length
        super(SphinxAdapter, self).__init__(reason=reason, version=version, action=action, category=category)

    def __call__(self, wrapped):
        """
        Add the Sphinx directive to your class or function.

        :param wrapped: Wrapped class or function.

        :return: the decorated class or function.
        """
        fmt = '.. {directive}:: {version}' if self.version else '.. {directive}::'
        div_lines = [fmt.format(directive=self.directive, version=self.version)]
        width = self.line_length - 3 if self.line_length > 3 else 2 ** 16
        reason = textwrap.dedent(self.reason).strip()
        for paragraph in reason.splitlines():
            if paragraph:
                div_lines.extend(textwrap.fill(paragraph, width=width, initial_indent='   ', subsequent_indent='   ').splitlines())
            else:
                div_lines.append('')
        docstring = wrapped.__doc__ or ''
        lines = docstring.splitlines(keepends=True) or ['']
        docstring = textwrap.dedent(''.join(lines[1:])) if len(lines) > 1 else ''
        docstring = lines[0] + docstring
        if docstring:
            docstring = re.sub('\\n+$', '', docstring, flags=re.DOTALL) + '\n\n'
        else:
            docstring = '\n'
        docstring += ''.join(('{}\n'.format(line) for line in div_lines))
        wrapped.__doc__ = docstring
        if self.directive in {'versionadded', 'versionchanged'}:
            return wrapped
        return super(SphinxAdapter, self).__call__(wrapped)

    def get_deprecated_msg(self, wrapped, instance):
        """
        Get the deprecation warning message (without Sphinx cross-referencing syntax) for the user.

        :param wrapped: Wrapped class or function.

        :param instance: The object to which the wrapped function was bound when it was called.

        :return: The warning message.

        .. versionadded:: 1.2.12
           Strip Sphinx cross-referencing syntax from warning message.

        """
        msg = super(SphinxAdapter, self).get_deprecated_msg(wrapped, instance)
        msg = re.sub('(?: : [a-zA-Z]+ )? : [a-zA-Z]+ : (`[^`]*`)', '\\1', msg, flags=re.X)
        return msg