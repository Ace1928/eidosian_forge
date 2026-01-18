import re
from pygments.lexer import bygroups, default, inherit, words
from pygments.lexers.lisp import SchemeLexer
from pygments.lexers._lilypond_builtins import (
from pygments.token import Token
class LilyPondLexer(SchemeLexer):
    """
    Lexer for input to LilyPond, a text-based music typesetter.

    .. important::

       This lexer is meant to be used in conjunction with the ``lilypond`` style.

    .. versionadded:: 2.11
    """
    name = 'LilyPond'
    url = 'https://lilypond.org'
    aliases = ['lilypond']
    filenames = ['*.ly']
    mimetypes = []
    flags = re.DOTALL | re.MULTILINE

    def get_tokens_unprocessed(self, text):
        """Highlight Scheme variables as LilyPond builtins when applicable."""
        for index, token, value in super().get_tokens_unprocessed(text):
            if token is Token.Name.Function or token is Token.Name.Variable:
                if value in scheme_functions:
                    token = Token.Name.Builtin.SchemeFunction
            elif token is Token.Name.Builtin:
                token = Token.Name.Builtin.SchemeBuiltin
            yield (index, token, value)
    tokens = {'root': [('\\s+', Token.Text.Whitespace), ('%\\{.*?%\\}', Token.Comment.Multiline), ('%.*?$', Token.Comment.Single), ('#\\}', Token.Punctuation, '#pop'), ('[#$]@?', Token.Punctuation, 'value'), ('(?x)\n               \\\\\\\\\n               | (?<= \\s ) (?: -- | __ )\n               | [{}<>=.,:|]\n              ', Token.Punctuation), (words(pitches, suffix="=?[',]*!?\\??" + NAME_END_RE), Token.Pitch), ('[\\-_^]?"', Token.String, 'string'), ('-?\\d+\\.\\d+', Token.Number.Float), ('-?\\d+/\\d+', Token.Number.Fraction), ('(?x)\n               (?<= \\s ) -\\d+\n               | (?: (?: \\d+ | \\\\breve | \\\\longa | \\\\maxima )\n                     \\.* )\n              ', Token.Number), ('\\*', Token.Number), ('[~()[\\]]', Token.Name.Builtin.Articulation), ('[\\-_^][>^_!.\\-+]', Token.Name.Builtin.Articulation), ('[\\-_^]?\\\\?\\d+', Token.Name.Builtin.Articulation), (builtin_words(keywords, 'mandatory'), Token.Keyword), (builtin_words(pitch_language_names, 'disallowed'), Token.Name.PitchLanguage), (builtin_words(clefs, 'disallowed'), Token.Name.Builtin.Clef), (builtin_words(scales, 'mandatory'), Token.Name.Builtin.Scale), (builtin_words(repeat_types, 'disallowed'), Token.Name.Builtin.RepeatType), (builtin_words(units, 'mandatory'), Token.Number), (builtin_words(chord_modifiers, 'disallowed'), Token.ChordModifier), (builtin_words(music_functions, 'mandatory'), Token.Name.Builtin.MusicFunction), (builtin_words(dynamics, 'mandatory'), Token.Name.Builtin.Dynamic), (builtin_words(articulations, 'mandatory'), Token.Name.Builtin.Articulation), (builtin_words(music_commands, 'mandatory'), Token.Name.Builtin.MusicCommand), (builtin_words(markup_commands, 'mandatory'), Token.Name.Builtin.MarkupCommand), (builtin_words(grobs, 'disallowed'), Token.Name.Builtin.Grob), (builtin_words(translators, 'disallowed'), Token.Name.Builtin.Translator), (builtin_words(contexts, 'optional'), Token.Name.Builtin.Context), (builtin_words(context_properties, 'disallowed'), Token.Name.Builtin.ContextProperty), (builtin_words(grob_properties, 'disallowed'), Token.Name.Builtin.GrobProperty, 'maybe-subproperties'), (builtin_words(paper_variables, 'optional'), Token.Name.Builtin.PaperVariable), (builtin_words(header_variables, 'optional'), Token.Name.Builtin.HeaderVariable), ('[\\-_^]?\\\\.+?' + NAME_END_RE, Token.Name.BackslashReference), ('(?x)\n               (?: [^\\W\\d] | - )+\n               (?= (?: [^\\W\\d] | [\\-.] )* \\s* = )\n              ', Token.Name.Lvalue), ('([^\\W\\d]|-)+?' + NAME_END_RE, Token.Text), ('.', Token.Text)], 'string': [('"', Token.String, '#pop'), ('\\\\.', Token.String.Escape), ('[^\\\\"]+', Token.String)], 'value': [('#\\{', Token.Punctuation, ('#pop', 'root')), inherit], 'maybe-subproperties': [('\\s+', Token.Text.Whitespace), ('(\\.)((?:[^\\W\\d]|-)+?)' + NAME_END_RE, bygroups(Token.Punctuation, Token.Name.Builtin.GrobProperty)), default('#pop')]}