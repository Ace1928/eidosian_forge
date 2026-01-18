class NoThemeError(Error):
    """Raised when trying to access a nonexistant icon theme.
    
    The name of the theme is the .theme attribute.
    """

    def __init__(self, theme):
        Error.__init__(self, 'No such icon-theme: %s' % theme)
        self.theme = theme