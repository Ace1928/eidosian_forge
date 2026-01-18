import pprint
import re
import typing as t
from markupsafe import Markup
from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context
class InternationalizationExtension(Extension):
    """This extension adds gettext support to Jinja."""
    tags = {'trans'}

    def __init__(self, environment: Environment) -> None:
        super().__init__(environment)
        environment.globals['_'] = _gettext_alias
        environment.extend(install_gettext_translations=self._install, install_null_translations=self._install_null, install_gettext_callables=self._install_callables, uninstall_gettext_translations=self._uninstall, extract_translations=self._extract, newstyle_gettext=False)

    def _install(self, translations: '_SupportedTranslations', newstyle: t.Optional[bool]=None) -> None:
        gettext = getattr(translations, 'ugettext', None)
        if gettext is None:
            gettext = translations.gettext
        ngettext = getattr(translations, 'ungettext', None)
        if ngettext is None:
            ngettext = translations.ngettext
        pgettext = getattr(translations, 'pgettext', None)
        npgettext = getattr(translations, 'npgettext', None)
        self._install_callables(gettext, ngettext, newstyle=newstyle, pgettext=pgettext, npgettext=npgettext)

    def _install_null(self, newstyle: t.Optional[bool]=None) -> None:
        import gettext
        translations = gettext.NullTranslations()
        if hasattr(translations, 'pgettext'):
            pgettext = translations.pgettext
        else:

            def pgettext(c: str, s: str) -> str:
                return s
        if hasattr(translations, 'npgettext'):
            npgettext = translations.npgettext
        else:

            def npgettext(c: str, s: str, p: str, n: int) -> str:
                return s if n == 1 else p
        self._install_callables(gettext=translations.gettext, ngettext=translations.ngettext, newstyle=newstyle, pgettext=pgettext, npgettext=npgettext)

    def _install_callables(self, gettext: t.Callable[[str], str], ngettext: t.Callable[[str, str, int], str], newstyle: t.Optional[bool]=None, pgettext: t.Optional[t.Callable[[str, str], str]]=None, npgettext: t.Optional[t.Callable[[str, str, str, int], str]]=None) -> None:
        if newstyle is not None:
            self.environment.newstyle_gettext = newstyle
        if self.environment.newstyle_gettext:
            gettext = _make_new_gettext(gettext)
            ngettext = _make_new_ngettext(ngettext)
            if pgettext is not None:
                pgettext = _make_new_pgettext(pgettext)
            if npgettext is not None:
                npgettext = _make_new_npgettext(npgettext)
        self.environment.globals.update(gettext=gettext, ngettext=ngettext, pgettext=pgettext, npgettext=npgettext)

    def _uninstall(self, translations: '_SupportedTranslations') -> None:
        for key in ('gettext', 'ngettext', 'pgettext', 'npgettext'):
            self.environment.globals.pop(key, None)

    def _extract(self, source: t.Union[str, nodes.Template], gettext_functions: t.Sequence[str]=GETTEXT_FUNCTIONS) -> t.Iterator[t.Tuple[int, str, t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]]]]:
        if isinstance(source, str):
            source = self.environment.parse(source)
        return extract_from_ast(source, gettext_functions)

    def parse(self, parser: 'Parser') -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """Parse a translatable tag."""
        lineno = next(parser.stream).lineno
        context = None
        context_token = parser.stream.next_if('string')
        if context_token is not None:
            context = context_token.value
        plural_expr: t.Optional[nodes.Expr] = None
        plural_expr_assignment: t.Optional[nodes.Assign] = None
        num_called_num = False
        variables: t.Dict[str, nodes.Expr] = {}
        trimmed = None
        while parser.stream.current.type != 'block_end':
            if variables:
                parser.stream.expect('comma')
            if parser.stream.skip_if('colon'):
                break
            token = parser.stream.expect('name')
            if token.value in variables:
                parser.fail(f'translatable variable {token.value!r} defined twice.', token.lineno, exc=TemplateAssertionError)
            if parser.stream.current.type == 'assign':
                next(parser.stream)
                variables[token.value] = var = parser.parse_expression()
            elif trimmed is None and token.value in ('trimmed', 'notrimmed'):
                trimmed = token.value == 'trimmed'
                continue
            else:
                variables[token.value] = var = nodes.Name(token.value, 'load')
            if plural_expr is None:
                if isinstance(var, nodes.Call):
                    plural_expr = nodes.Name('_trans', 'load')
                    variables[token.value] = plural_expr
                    plural_expr_assignment = nodes.Assign(nodes.Name('_trans', 'store'), var)
                else:
                    plural_expr = var
                num_called_num = token.value == 'num'
        parser.stream.expect('block_end')
        plural = None
        have_plural = False
        referenced = set()
        singular_names, singular = self._parse_block(parser, True)
        if singular_names:
            referenced.update(singular_names)
            if plural_expr is None:
                plural_expr = nodes.Name(singular_names[0], 'load')
                num_called_num = singular_names[0] == 'num'
        if parser.stream.current.test('name:pluralize'):
            have_plural = True
            next(parser.stream)
            if parser.stream.current.type != 'block_end':
                token = parser.stream.expect('name')
                if token.value not in variables:
                    parser.fail(f'unknown variable {token.value!r} for pluralization', token.lineno, exc=TemplateAssertionError)
                plural_expr = variables[token.value]
                num_called_num = token.value == 'num'
            parser.stream.expect('block_end')
            plural_names, plural = self._parse_block(parser, False)
            next(parser.stream)
            referenced.update(plural_names)
        else:
            next(parser.stream)
        for name in referenced:
            if name not in variables:
                variables[name] = nodes.Name(name, 'load')
        if not have_plural:
            plural_expr = None
        elif plural_expr is None:
            parser.fail('pluralize without variables', lineno)
        if trimmed is None:
            trimmed = self.environment.policies['ext.i18n.trimmed']
        if trimmed:
            singular = self._trim_whitespace(singular)
            if plural:
                plural = self._trim_whitespace(plural)
        node = self._make_node(singular, plural, context, variables, plural_expr, bool(referenced), num_called_num and have_plural)
        node.set_lineno(lineno)
        if plural_expr_assignment is not None:
            return [plural_expr_assignment, node]
        else:
            return node

    def _trim_whitespace(self, string: str, _ws_re: t.Pattern[str]=_ws_re) -> str:
        return _ws_re.sub(' ', string.strip())

    def _parse_block(self, parser: 'Parser', allow_pluralize: bool) -> t.Tuple[t.List[str], str]:
        """Parse until the next block tag with a given name."""
        referenced = []
        buf = []
        while True:
            if parser.stream.current.type == 'data':
                buf.append(parser.stream.current.value.replace('%', '%%'))
                next(parser.stream)
            elif parser.stream.current.type == 'variable_begin':
                next(parser.stream)
                name = parser.stream.expect('name').value
                referenced.append(name)
                buf.append(f'%({name})s')
                parser.stream.expect('variable_end')
            elif parser.stream.current.type == 'block_begin':
                next(parser.stream)
                block_name = parser.stream.current.value if parser.stream.current.type == 'name' else None
                if block_name == 'endtrans':
                    break
                elif block_name == 'pluralize':
                    if allow_pluralize:
                        break
                    parser.fail('a translatable section can have only one pluralize section')
                elif block_name == 'trans':
                    parser.fail("trans blocks can't be nested; did you mean `endtrans`?")
                parser.fail(f'control structures in translatable sections are not allowed; saw `{block_name}`')
            elif parser.stream.eos:
                parser.fail('unclosed translation block')
            else:
                raise RuntimeError('internal parser error')
        return (referenced, concat(buf))

    def _make_node(self, singular: str, plural: t.Optional[str], context: t.Optional[str], variables: t.Dict[str, nodes.Expr], plural_expr: t.Optional[nodes.Expr], vars_referenced: bool, num_called_num: bool) -> nodes.Output:
        """Generates a useful node from the data provided."""
        newstyle = self.environment.newstyle_gettext
        node: nodes.Expr
        if not vars_referenced and (not newstyle):
            singular = singular.replace('%%', '%')
            if plural:
                plural = plural.replace('%%', '%')
        func_name = 'gettext'
        func_args: t.List[nodes.Expr] = [nodes.Const(singular)]
        if context is not None:
            func_args.insert(0, nodes.Const(context))
            func_name = f'p{func_name}'
        if plural_expr is not None:
            func_name = f'n{func_name}'
            func_args.extend((nodes.Const(plural), plural_expr))
        node = nodes.Call(nodes.Name(func_name, 'load'), func_args, [], None, None)
        if newstyle:
            for key, value in variables.items():
                if num_called_num and key == 'num':
                    continue
                node.kwargs.append(nodes.Keyword(key, value))
        else:
            node = nodes.MarkSafeIfAutoescape(node)
            if variables:
                node = nodes.Mod(node, nodes.Dict([nodes.Pair(nodes.Const(key), value) for key, value in variables.items()]))
        return nodes.Output([node])