from gast import AST  # so that metadata are walkable as regular ast nodes
class Comprehension(AST):

    def __init__(self, *args):
        super(Comprehension, self).__init__()
        if args:
            self.target = args[0]