class PublicBranchOutOfDate(BzrError):
    _fmt = 'Public branch "%(public_location)s" lacks revision "%(revstring)s".'

    def __init__(self, public_location, revstring):
        import breezy.urlutils as urlutils
        public_location = urlutils.unescape_for_display(public_location, 'ascii')
        BzrError.__init__(self, public_location=public_location, revstring=revstring)