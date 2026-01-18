class TagAlreadyExists(BzrError):
    _fmt = 'Tag %(tag_name)s already exists.'

    def __init__(self, tag_name):
        self.tag_name = tag_name