import pygments
import pygments.formatter
class OdtPygmentsLaTeXFormatter(OdtPygmentsFormatter):

    def format(self, tokensource, outfile):
        tokenclass = pygments.token.Token
        for ttype, value in tokensource:
            value = self.escape_function(value)
            if ttype == tokenclass.Keyword:
                s2 = self.rststyle('codeblock-keyword')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype in (tokenclass.Literal.String, tokenclass.Literal.String.Backtick):
                s2 = self.rststyle('codeblock-string')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype == tokenclass.Name.Attribute:
                s2 = self.rststyle('codeblock-operator')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype == tokenclass.Comment:
                if value[-1] == '\n':
                    s2 = self.rststyle('codeblock-comment')
                    s1 = '<text:span text:style-name="%s">%s</text:span>\n' % (s2, value[:-1])
                else:
                    s2 = self.rststyle('codeblock-comment')
                    s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            elif ttype == tokenclass.Name.Builtin:
                s2 = self.rststyle('codeblock-name')
                s1 = '<text:span text:style-name="%s">%s</text:span>' % (s2, value)
            else:
                s1 = value
            outfile.write(s1)