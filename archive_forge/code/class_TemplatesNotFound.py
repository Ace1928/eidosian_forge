import typing as t
class TemplatesNotFound(TemplateNotFound):
    """Like :class:`TemplateNotFound` but raised if multiple templates
    are selected.  This is a subclass of :class:`TemplateNotFound`
    exception, so just catching the base exception will catch both.

    .. versionchanged:: 2.11
        If a name in the list of names is :class:`Undefined`, a message
        about it being undefined is shown rather than the empty string.

    .. versionadded:: 2.2
    """

    def __init__(self, names: t.Sequence[t.Union[str, 'Undefined']]=(), message: t.Optional[str]=None) -> None:
        if message is None:
            from .runtime import Undefined
            parts = []
            for name in names:
                if isinstance(name, Undefined):
                    parts.append(name._undefined_message)
                else:
                    parts.append(name)
            parts_str = ', '.join(map(str, parts))
            message = f'none of the templates given were found: {parts_str}'
        super().__init__(names[-1] if names else None, message)
        self.templates = list(names)