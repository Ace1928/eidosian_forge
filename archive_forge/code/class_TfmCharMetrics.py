class TfmCharMetrics(object):

    def __init__(self, width, height, depth, italic, kern_table):
        self.width = width
        self.height = height
        self.depth = depth
        self.italic_correction = italic
        self.kern_table = kern_table