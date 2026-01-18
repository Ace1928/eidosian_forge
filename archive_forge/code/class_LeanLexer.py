import re
from pygments.lexer import RegexLexer, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class LeanLexer(RegexLexer):
    """
    For the `Lean <https://github.com/leanprover/lean>`_
    theorem prover.

    .. versionadded:: 2.0
    """
    name = 'Lean'
    aliases = ['lean']
    filenames = ['*.lean']
    mimetypes = ['text/x-lean']
    flags = re.MULTILINE | re.UNICODE
    keywords1 = ('import', 'abbreviation', 'opaque_hint', 'tactic_hint', 'definition', 'renaming', 'inline', 'hiding', 'exposing', 'parameter', 'parameters', 'conjecture', 'hypothesis', 'lemma', 'corollary', 'variable', 'variables', 'theorem', 'axiom', 'inductive', 'structure', 'universe', 'alias', 'help', 'options', 'precedence', 'postfix', 'prefix', 'calc_trans', 'calc_subst', 'calc_refl', 'infix', 'infixl', 'infixr', 'notation', 'eval', 'check', 'exit', 'coercion', 'end', 'private', 'using', 'namespace', 'including', 'instance', 'section', 'context', 'protected', 'expose', 'export', 'set_option', 'add_rewrite', 'extends', 'open', 'example', 'constant', 'constants', 'print', 'opaque', 'reducible', 'irreducible')
    keywords2 = ('forall', 'fun', 'Pi', 'obtain', 'from', 'have', 'show', 'assume', 'take', 'let', 'if', 'else', 'then', 'by', 'in', 'with', 'begin', 'proof', 'qed', 'calc', 'match')
    keywords3 = ('Type', 'Prop')
    operators = (u'!=', u'#', u'&', u'&&', u'*', u'+', u'-', u'/', u'@', u'!', u'`', u'-.', u'->', u'.', u'..', u'...', u'::', u':>', u';', u';;', u'<', u'<-', u'=', u'==', u'>', u'_', u'|', u'||', u'~', u'=>', u'<=', u'>=', u'/\\', u'\\/', u'∀', u'Π', u'λ', u'↔', u'∧', u'∨', u'≠', u'≤', u'≥', u'¬', u'⁻¹', u'⬝', u'▸', u'→', u'∃', u'ℕ', u'ℤ', u'≈', u'×', u'⌞', u'⌟', u'≡', u'⟨', u'⟩')
    punctuation = (u'(', u')', u':', u'{', u'}', u'[', u']', u'⦃', u'⦄', u':=', u',')
    tokens = {'root': [('\\s+', Text), ('/-', Comment, 'comment'), ('--.*?$', Comment.Single), (words(keywords1, prefix='\\b', suffix='\\b'), Keyword.Namespace), (words(keywords2, prefix='\\b', suffix='\\b'), Keyword), (words(keywords3, prefix='\\b', suffix='\\b'), Keyword.Type), (words(operators), Name.Builtin.Pseudo), (words(punctuation), Operator), (u"[A-Za-z_α-κμ-ϻἀ-῾℀-⅏][A-Za-z_'α-κμ-ϻἀ-῾⁰-⁹ⁿ-₉ₐ-ₜ℀-⅏0-9]*", Name), ('\\d+', Number.Integer), ('"', String.Double, 'string'), ("[~?][a-z][\\w\\']*:", Name.Variable)], 'comment': [('[^/-]', Comment.Multiline), ('/-', Comment.Multiline, '#push'), ('-/', Comment.Multiline, '#pop'), ('[/-]', Comment.Multiline)], 'string': [('[^\\\\"]+', String.Double), ('\\\\[n"\\\\]', String.Escape), ('"', String.Double, '#pop')]}