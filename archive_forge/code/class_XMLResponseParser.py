import json, sys
from xml.dom import minidom
import plistlib
import logging
class XMLResponseParser(ResponseParser):

    def __init__(self):
        try:
            import libxml2
        except ImportError:
            print('libxml2 XMLResponseParser requires libxml2')
            sys.exit(1)
        self.meta = 'text/xml'

    def parse(self, xml, pvars):
        import libxml2
        doc = libxml2.parseDoc(xml)
        pvars = self.getVars(pvars)
        vals = {}
        for k, v in pvars.items():
            res = doc.xpathEval(v)
            vals[k] = []
            for r in res:
                if r.type == 'element':
                    vals[k].append(self.xmlToDict(minidom.parseString(str(r)))[r.name])
                elif r.type == 'attribute':
                    vals[k].append(r.content)
                else:
                    logger.error('UNKNOWN TYPE')
            if len(vals[k]) == 1:
                vals[k] = vals[k][0]
            elif len(vals[k]) == 0:
                vals[k] = None
        return vals

    def xmlToDict(self, xmlNode):
        if xmlNode.nodeName == '#document':
            node = {xmlNode.firstChild.nodeName: {}}
            node[xmlNode.firstChild.nodeName] = self.xmlToDict(xmlNode.firstChild)
            return node
        node = {}
        curr = node
        if xmlNode.attributes:
            for name, value in xmlNode.attributes.items():
                curr[name] = value
        for n in xmlNode.childNodes:
            if n.nodeType == n.TEXT_NODE:
                curr['__TEXT__'] = n.data
                continue
            if not n.nodeName in curr:
                curr[n.nodeName] = []
            if len(xmlNode.getElementsByTagName(n.nodeName)) > 1:
                curr[n.nodeName].append(self.xmlToDict(n))
            else:
                curr[n.nodeName] = self.xmlToDict(n)
        return node