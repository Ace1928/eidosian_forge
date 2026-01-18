import enum
class ProblemFormat(str, enum.Enum):
    pyomo = 'pyomo'
    cpxlp = 'cpxlp'
    nl = 'nl'
    mps = 'mps'
    mod = 'mod'
    lpxlp = 'lpxlp'
    osil = 'osil'
    bar = 'bar'
    gams = 'gams'

    def __str__(self):
        return self.value