import re
from pygments.lexer import Lexer, RegexLexer, ExtendedRegexLexer, include, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
class RubyLexer(ExtendedRegexLexer):
    """
    For `Ruby <http://www.ruby-lang.org>`_ source code.
    """
    name = 'Ruby'
    aliases = ['rb', 'ruby', 'duby']
    filenames = ['*.rb', '*.rbw', 'Rakefile', '*.rake', '*.gemspec', '*.rbx', '*.duby', 'Gemfile']
    mimetypes = ['text/x-ruby', 'application/x-ruby']
    flags = re.DOTALL | re.MULTILINE

    def heredoc_callback(self, match, ctx):
        start = match.start(1)
        yield (start, Operator, match.group(1))
        yield (match.start(2), String.Heredoc, match.group(2))
        yield (match.start(3), String.Delimiter, match.group(3))
        yield (match.start(4), String.Heredoc, match.group(4))
        heredocstack = ctx.__dict__.setdefault('heredocstack', [])
        outermost = not bool(heredocstack)
        heredocstack.append((match.group(1) == '<<-', match.group(3)))
        ctx.pos = match.start(5)
        ctx.end = match.end(5)
        for i, t, v in self.get_tokens_unprocessed(context=ctx):
            yield (i, t, v)
        ctx.pos = match.end()
        if outermost:
            for tolerant, hdname in heredocstack:
                lines = []
                for match in line_re.finditer(ctx.text, ctx.pos):
                    if tolerant:
                        check = match.group().strip()
                    else:
                        check = match.group().rstrip()
                    if check == hdname:
                        for amatch in lines:
                            yield (amatch.start(), String.Heredoc, amatch.group())
                        yield (match.start(), String.Delimiter, match.group())
                        ctx.pos = match.end()
                        break
                    else:
                        lines.append(match)
                else:
                    for amatch in lines:
                        yield (amatch.start(), Error, amatch.group())
            ctx.end = len(ctx.text)
            del heredocstack[:]

    def gen_rubystrings_rules():

        def intp_regex_callback(self, match, ctx):
            yield (match.start(1), String.Regex, match.group(1))
            nctx = LexerContext(match.group(3), 0, ['interpolated-regex'])
            for i, t, v in self.get_tokens_unprocessed(context=nctx):
                yield (match.start(3) + i, t, v)
            yield (match.start(4), String.Regex, match.group(4))
            ctx.pos = match.end()

        def intp_string_callback(self, match, ctx):
            yield (match.start(1), String.Other, match.group(1))
            nctx = LexerContext(match.group(3), 0, ['interpolated-string'])
            for i, t, v in self.get_tokens_unprocessed(context=nctx):
                yield (match.start(3) + i, t, v)
            yield (match.start(4), String.Other, match.group(4))
            ctx.pos = match.end()
        states = {}
        states['strings'] = [('\\:@{0,2}[a-zA-Z_]\\w*[!?]?', String.Symbol), (words(RUBY_OPERATORS, prefix='\\:@{0,2}'), String.Symbol), (":'(\\\\\\\\|\\\\'|[^'])*'", String.Symbol), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single), (':"', String.Symbol, 'simple-sym'), ('([a-zA-Z_]\\w*)(:)(?!:)', bygroups(String.Symbol, Punctuation)), ('"', String.Double, 'simple-string'), ('(?<!\\.)`', String.Backtick, 'simple-backtick')]
        for name, ttype, end in (('string', String.Double, '"'), ('sym', String.Symbol, '"'), ('backtick', String.Backtick, '`')):
            states['simple-' + name] = [include('string-intp-escaped'), ('[^\\\\%s#]+' % end, ttype), ('[\\\\#]', ttype), (end, ttype, '#pop')]
        for lbrace, rbrace, bracecc, name in (('\\{', '\\}', '{}', 'cb'), ('\\[', '\\]', '\\[\\]', 'sb'), ('\\(', '\\)', '()', 'pa'), ('<', '>', '<>', 'ab')):
            states[name + '-intp-string'] = [('\\\\[\\\\' + bracecc + ']', String.Other), (lbrace, String.Other, '#push'), (rbrace, String.Other, '#pop'), include('string-intp-escaped'), ('[\\\\#' + bracecc + ']', String.Other), ('[^\\\\#' + bracecc + ']+', String.Other)]
            states['strings'].append(('%[QWx]?' + lbrace, String.Other, name + '-intp-string'))
            states[name + '-string'] = [('\\\\[\\\\' + bracecc + ']', String.Other), (lbrace, String.Other, '#push'), (rbrace, String.Other, '#pop'), ('[\\\\#' + bracecc + ']', String.Other), ('[^\\\\#' + bracecc + ']+', String.Other)]
            states['strings'].append(('%[qsw]' + lbrace, String.Other, name + '-string'))
            states[name + '-regex'] = [('\\\\[\\\\' + bracecc + ']', String.Regex), (lbrace, String.Regex, '#push'), (rbrace + '[mixounse]*', String.Regex, '#pop'), include('string-intp'), ('[\\\\#' + bracecc + ']', String.Regex), ('[^\\\\#' + bracecc + ']+', String.Regex)]
            states['strings'].append(('%r' + lbrace, String.Regex, name + '-regex'))
        states['strings'] += [('(%r([\\W_]))((?:\\\\\\2|(?!\\2).)*)(\\2[mixounse]*)', intp_regex_callback), ('%[qsw]([\\W_])((?:\\\\\\1|(?!\\1).)*)\\1', String.Other), ('(%[QWx]([\\W_]))((?:\\\\\\2|(?!\\2).)*)(\\2)', intp_string_callback), ('(?<=[-+/*%=<>&!^|~,(])(\\s*)(%([\\t ])(?:(?:\\\\\\3|(?!\\3).)*)\\3)', bygroups(Text, String.Other, None)), ('^(\\s*)(%([\\t ])(?:(?:\\\\\\3|(?!\\3).)*)\\3)', bygroups(Text, String.Other, None)), ('(%([^a-zA-Z0-9\\s]))((?:\\\\\\2|(?!\\2).)*)(\\2)', intp_string_callback)]
        return states
    tokens = {'root': [('\\A#!.+?$', Comment.Hashbang), ('#.*?$', Comment.Single), ('=begin\\s.*?\\n=end.*?$', Comment.Multiline), (words(('BEGIN', 'END', 'alias', 'begin', 'break', 'case', 'defined?', 'do', 'else', 'elsif', 'end', 'ensure', 'for', 'if', 'in', 'next', 'redo', 'rescue', 'raise', 'retry', 'return', 'super', 'then', 'undef', 'unless', 'until', 'when', 'while', 'yield'), suffix='\\b'), Keyword), ('(module)(\\s+)([a-zA-Z_]\\w*(?:::[a-zA-Z_]\\w*)*)', bygroups(Keyword, Text, Name.Namespace)), ('(def)(\\s+)', bygroups(Keyword, Text), 'funcname'), ('def(?=[*%&^`~+-/\\[<>=])', Keyword, 'funcname'), ('(class)(\\s+)', bygroups(Keyword, Text), 'classname'), (words(('initialize', 'new', 'loop', 'include', 'extend', 'raise', 'attr_reader', 'attr_writer', 'attr_accessor', 'attr', 'catch', 'throw', 'private', 'module_function', 'public', 'protected', 'true', 'false', 'nil'), suffix='\\b'), Keyword.Pseudo), ('(not|and|or)\\b', Operator.Word), (words(('autoload', 'block_given', 'const_defined', 'eql', 'equal', 'frozen', 'include', 'instance_of', 'is_a', 'iterator', 'kind_of', 'method_defined', 'nil', 'private_method_defined', 'protected_method_defined', 'public_method_defined', 'respond_to', 'tainted'), suffix='\\?'), Name.Builtin), ('(chomp|chop|exit|gsub|sub)!', Name.Builtin), (words(('Array', 'Float', 'Integer', 'String', '__id__', '__send__', 'abort', 'ancestors', 'at_exit', 'autoload', 'binding', 'callcc', 'caller', 'catch', 'chomp', 'chop', 'class_eval', 'class_variables', 'clone', 'const_defined?', 'const_get', 'const_missing', 'const_set', 'constants', 'display', 'dup', 'eval', 'exec', 'exit', 'extend', 'fail', 'fork', 'format', 'freeze', 'getc', 'gets', 'global_variables', 'gsub', 'hash', 'id', 'included_modules', 'inspect', 'instance_eval', 'instance_method', 'instance_methods', 'instance_variable_get', 'instance_variable_set', 'instance_variables', 'lambda', 'load', 'local_variables', 'loop', 'method', 'method_missing', 'methods', 'module_eval', 'name', 'object_id', 'open', 'p', 'print', 'printf', 'private_class_method', 'private_instance_methods', 'private_methods', 'proc', 'protected_instance_methods', 'protected_methods', 'public_class_method', 'public_instance_methods', 'public_methods', 'putc', 'puts', 'raise', 'rand', 'readline', 'readlines', 'require', 'scan', 'select', 'self', 'send', 'set_trace_func', 'singleton_methods', 'sleep', 'split', 'sprintf', 'srand', 'sub', 'syscall', 'system', 'taint', 'test', 'throw', 'to_a', 'to_s', 'trace_var', 'trap', 'untaint', 'untrace_var', 'warn'), prefix='(?<!\\.)', suffix='\\b'), Name.Builtin), ('__(FILE|LINE)__\\b', Name.Builtin.Pseudo), ('(?<!\\w)(<<-?)(["`\\\']?)([a-zA-Z_]\\w*)(\\2)(.*?\\n)', heredoc_callback), ('(<<-?)("|\\\')()(\\2)(.*?\\n)', heredoc_callback), ('__END__', Comment.Preproc, 'end-part'), ('(?:^|(?<=[=<>~!:])|(?<=(?:\\s|;)when\\s)|(?<=(?:\\s|;)or\\s)|(?<=(?:\\s|;)and\\s)|(?<=\\.index\\s)|(?<=\\.scan\\s)|(?<=\\.sub\\s)|(?<=\\.sub!\\s)|(?<=\\.gsub\\s)|(?<=\\.gsub!\\s)|(?<=\\.match\\s)|(?<=(?:\\s|;)if\\s)|(?<=(?:\\s|;)elsif\\s)|(?<=^when\\s)|(?<=^index\\s)|(?<=^scan\\s)|(?<=^sub\\s)|(?<=^gsub\\s)|(?<=^sub!\\s)|(?<=^gsub!\\s)|(?<=^match\\s)|(?<=^if\\s)|(?<=^elsif\\s))(\\s*)(/)', bygroups(Text, String.Regex), 'multiline-regex'), ('(?<=\\(|,|\\[)/', String.Regex, 'multiline-regex'), ('(\\s+)(/)(?![\\s=])', bygroups(Text, String.Regex), 'multiline-regex'), ('(0_?[0-7]+(?:_[0-7]+)*)(\\s*)([/?])?', bygroups(Number.Oct, Text, Operator)), ('(0x[0-9A-Fa-f]+(?:_[0-9A-Fa-f]+)*)(\\s*)([/?])?', bygroups(Number.Hex, Text, Operator)), ('(0b[01]+(?:_[01]+)*)(\\s*)([/?])?', bygroups(Number.Bin, Text, Operator)), ('([\\d]+(?:_\\d+)*)(\\s*)([/?])?', bygroups(Number.Integer, Text, Operator)), ('@@[a-zA-Z_]\\w*', Name.Variable.Class), ('@[a-zA-Z_]\\w*', Name.Variable.Instance), ('\\$\\w+', Name.Variable.Global), ('\\$[!@&`\\\'+~=/\\\\,;.<>_*$?:"^-]', Name.Variable.Global), ('\\$-[0adFiIlpvw]', Name.Variable.Global), ('::', Operator), include('strings'), ('\\?(\\\\[MC]-)*(\\\\([\\\\abefnrstv#"\\\']|x[a-fA-F0-9]{1,2}|[0-7]{1,3})|\\S)(?!\\w)', String.Char), ('[A-Z]\\w+', Name.Constant), (words(RUBY_OPERATORS, prefix='(\\.|::)'), bygroups(Operator, Name.Operator)), ('(\\.|::)([a-zA-Z_]\\w*[!?]?|[*%&^`~+\\-/\\[<>=])', bygroups(Operator, Name)), ('[a-zA-Z_]\\w*[!?]?', Name), ('(\\[|\\]|\\*\\*|<<?|>>?|>=|<=|<=>|=~|={3}|!~|&&?|\\|\\||\\.{1,3})', Operator), ('[-+/*%=<>&!^|~]=?', Operator), ('[(){};,/?:\\\\]', Punctuation), ('\\s+', Text)], 'funcname': [('\\(', Punctuation, 'defexpr'), ('(?:([a-zA-Z_]\\w*)(\\.))?([a-zA-Z_]\\w*[!?]?|\\*\\*?|[-+]@?|[/%&|^`~]|\\[\\]=?|<<|>>|<=?>|>=?|===?)', bygroups(Name.Class, Operator, Name.Function), '#pop'), default('#pop')], 'classname': [('\\(', Punctuation, 'defexpr'), ('<<', Operator, '#pop'), ('[A-Z_]\\w*', Name.Class, '#pop'), default('#pop')], 'defexpr': [('(\\))(\\.|::)?', bygroups(Punctuation, Operator), '#pop'), ('\\(', Operator, '#push'), include('root')], 'in-intp': [('\\{', String.Interpol, '#push'), ('\\}', String.Interpol, '#pop'), include('root')], 'string-intp': [('#\\{', String.Interpol, 'in-intp'), ('#@@?[a-zA-Z_]\\w*', String.Interpol), ('#\\$[a-zA-Z_]\\w*', String.Interpol)], 'string-intp-escaped': [include('string-intp'), ('\\\\([\\\\abefnrstv#"\\\']|x[a-fA-F0-9]{1,2}|[0-7]{1,3})', String.Escape)], 'interpolated-regex': [include('string-intp'), ('[\\\\#]', String.Regex), ('[^\\\\#]+', String.Regex)], 'interpolated-string': [include('string-intp'), ('[\\\\#]', String.Other), ('[^\\\\#]+', String.Other)], 'multiline-regex': [include('string-intp'), ('\\\\\\\\', String.Regex), ('\\\\/', String.Regex), ('[\\\\#]', String.Regex), ('[^\\\\/#]+', String.Regex), ('/[mixounse]*', String.Regex, '#pop')], 'end-part': [('.+', Comment.Preproc, '#pop')]}
    tokens.update(gen_rubystrings_rules())

    def analyse_text(text):
        return shebang_matches(text, 'ruby(1\\.\\d)?')