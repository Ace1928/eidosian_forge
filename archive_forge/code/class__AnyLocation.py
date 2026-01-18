from io import StringIO
class _AnyLocation:

    def __init__(self):
        self.predicates = []
        self.elementName = None
        self.childLocation = None

    def matchesPredicates(self, elem):
        for p in self.predicates:
            if not p.value(elem):
                return 0
        return 1

    def listParents(self, elem, parentlist):
        if elem.parent != None:
            self.listParents(elem.parent, parentlist)
        parentlist.append(elem.name)

    def isRootMatch(self, elem):
        if (self.elementName == None or self.elementName == elem.name) and self.matchesPredicates(elem):
            if self.childLocation != None:
                for c in elem.elements():
                    if self.childLocation.matches(c):
                        return True
            else:
                return True
        return False

    def findFirstRootMatch(self, elem):
        if (self.elementName == None or self.elementName == elem.name) and self.matchesPredicates(elem):
            if self.childLocation != None:
                for c in elem.elements():
                    if self.childLocation.matches(c):
                        return c
                return None
            else:
                return elem
        else:
            for c in elem.elements():
                if self.matches(c):
                    return c
            return None

    def matches(self, elem):
        if self.isRootMatch(elem):
            return True
        else:
            for c in elem.elements():
                if self.matches(c):
                    return True
            return False

    def queryForString(self, elem, resultbuf):
        raise NotImplementedError('queryForString is not implemented for any location')

    def queryForNodes(self, elem, resultlist):
        if self.isRootMatch(elem):
            resultlist.append(elem)
        for c in elem.elements():
            self.queryForNodes(c, resultlist)

    def queryForStringList(self, elem, resultlist):
        if self.isRootMatch(elem):
            for c in elem.children:
                if isinstance(c, str):
                    resultlist.append(c)
        for c in elem.elements():
            self.queryForStringList(c, resultlist)