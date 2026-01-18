from io import StringIO
class XPathQuery:

    def __init__(self, queryStr):
        self.queryStr = queryStr
        from twisted.words.xish.xpathparser import XPathParser, XPathParserScanner
        parser = XPathParser(XPathParserScanner(queryStr))
        self.baseLocation = getattr(parser, 'XPATH')()

    def __hash__(self):
        return self.queryStr.__hash__()

    def matches(self, elem):
        return self.baseLocation.matches(elem)

    def queryForString(self, elem):
        result = StringIO()
        self.baseLocation.queryForString(elem, result)
        return result.getvalue()

    def queryForNodes(self, elem):
        result = []
        self.baseLocation.queryForNodes(elem, result)
        if len(result) == 0:
            return None
        else:
            return result

    def queryForStringList(self, elem):
        result = []
        self.baseLocation.queryForStringList(elem, result)
        if len(result) == 0:
            return None
        else:
            return result