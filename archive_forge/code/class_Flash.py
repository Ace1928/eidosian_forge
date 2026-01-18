import xml.sax.saxutils
class Flash(Application):

    def __init__(self, url, *args, **kwargs):
        self.url = url
        super(Flash, self).__init__(*args, **kwargs)

    def get_inner_content(self, content):
        content = OrderedContent()
        content.append_field('FlashMovieURL', self.url)
        super(Flash, self).get_inner_content(content)