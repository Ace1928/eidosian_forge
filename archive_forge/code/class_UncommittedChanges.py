class UncommittedChanges(BzrError):
    _fmt = 'Working tree "%(display_url)s" has uncommitted changes (See brz status).%(more)s'

    def __init__(self, tree, more=None):
        if more is None:
            more = ''
        else:
            more = ' ' + more
        import breezy.urlutils as urlutils
        user_url = getattr(tree, 'user_url', None)
        if user_url is None:
            display_url = str(tree)
        else:
            display_url = urlutils.unescape_for_display(user_url, 'ascii')
        BzrError.__init__(self, tree=tree, display_url=display_url, more=more)