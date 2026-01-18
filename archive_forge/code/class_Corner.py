from .simplex import SubsimplexName
class Corner:

    def __init__(self, tetrahedron, subsimplex):
        self.Tetrahedron = tetrahedron
        self.Subsimplex = subsimplex

    def __repr__(self):
        return '<' + SubsimplexName[self.Subsimplex] + ' of ' + str(self.Tetrahedron) + '>'