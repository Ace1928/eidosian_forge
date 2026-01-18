class RsrcDef(object):

    def __init__(self, properties, depends_on):
        self.properties = properties
        self.depends_on = depends_on

    def __repr__(self):
        return 'RsrcDef(%r, %r)' % (self.properties, self.depends_on)