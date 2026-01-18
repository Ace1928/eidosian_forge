class NoSuchTag(BzrError):
    _fmt = 'No such tag: %(tag_name)s'

    def __init__(self, tag_name):
        self.tag_name = tag_name