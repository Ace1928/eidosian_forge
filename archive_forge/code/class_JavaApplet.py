import xml.sax.saxutils
class JavaApplet(Application):

    def __init__(self, path, filename, *args, **kwargs):
        self.path = path
        self.filename = filename
        super(JavaApplet, self).__init__(*args, **kwargs)

    def get_inner_content(self, content):
        content = OrderedContent()
        content.append_field('AppletPath', self.path)
        content.append_field('AppletFilename', self.filename)
        super(JavaApplet, self).get_inner_content(content)