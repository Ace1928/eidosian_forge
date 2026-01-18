class EndToken(TokenBase):
    lbp = 0

    def nud(self, parser):
        raise parser.error_class('Unexpected end of expression in if tag.')