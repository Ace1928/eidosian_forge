from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
class InfosetFilter(object):
    replacementRegexp = re.compile('U[\\dA-F]{5,5}')

    def __init__(self, dropXmlnsLocalName=False, dropXmlnsAttrNs=False, preventDoubleDashComments=False, preventDashAtCommentEnd=False, replaceFormFeedCharacters=True, preventSingleQuotePubid=False):
        self.dropXmlnsLocalName = dropXmlnsLocalName
        self.dropXmlnsAttrNs = dropXmlnsAttrNs
        self.preventDoubleDashComments = preventDoubleDashComments
        self.preventDashAtCommentEnd = preventDashAtCommentEnd
        self.replaceFormFeedCharacters = replaceFormFeedCharacters
        self.preventSingleQuotePubid = preventSingleQuotePubid
        self.replaceCache = {}

    def coerceAttribute(self, name, namespace=None):
        if self.dropXmlnsLocalName and name.startswith('xmlns:'):
            warnings.warn('Attributes cannot begin with xmlns', DataLossWarning)
            return None
        elif self.dropXmlnsAttrNs and namespace == 'http://www.w3.org/2000/xmlns/':
            warnings.warn('Attributes cannot be in the xml namespace', DataLossWarning)
            return None
        else:
            return self.toXmlName(name)

    def coerceElement(self, name):
        return self.toXmlName(name)

    def coerceComment(self, data):
        if self.preventDoubleDashComments:
            while '--' in data:
                warnings.warn('Comments cannot contain adjacent dashes', DataLossWarning)
                data = data.replace('--', '- -')
            if data.endswith('-'):
                warnings.warn('Comments cannot end in a dash', DataLossWarning)
                data += ' '
        return data

    def coerceCharacters(self, data):
        if self.replaceFormFeedCharacters:
            for _ in range(data.count('\x0c')):
                warnings.warn('Text cannot contain U+000C', DataLossWarning)
            data = data.replace('\x0c', ' ')
        return data

    def coercePubid(self, data):
        dataOutput = data
        for char in nonPubidCharRegexp.findall(data):
            warnings.warn('Coercing non-XML pubid', DataLossWarning)
            replacement = self.getReplacementCharacter(char)
            dataOutput = dataOutput.replace(char, replacement)
        if self.preventSingleQuotePubid and dataOutput.find("'") >= 0:
            warnings.warn('Pubid cannot contain single quote', DataLossWarning)
            dataOutput = dataOutput.replace("'", self.getReplacementCharacter("'"))
        return dataOutput

    def toXmlName(self, name):
        nameFirst = name[0]
        nameRest = name[1:]
        m = nonXmlNameFirstBMPRegexp.match(nameFirst)
        if m:
            warnings.warn('Coercing non-XML name: %s' % name, DataLossWarning)
            nameFirstOutput = self.getReplacementCharacter(nameFirst)
        else:
            nameFirstOutput = nameFirst
        nameRestOutput = nameRest
        replaceChars = set(nonXmlNameBMPRegexp.findall(nameRest))
        for char in replaceChars:
            warnings.warn('Coercing non-XML name: %s' % name, DataLossWarning)
            replacement = self.getReplacementCharacter(char)
            nameRestOutput = nameRestOutput.replace(char, replacement)
        return nameFirstOutput + nameRestOutput

    def getReplacementCharacter(self, char):
        if char in self.replaceCache:
            replacement = self.replaceCache[char]
        else:
            replacement = self.escapeChar(char)
        return replacement

    def fromXmlName(self, name):
        for item in set(self.replacementRegexp.findall(name)):
            name = name.replace(item, self.unescapeChar(item))
        return name

    def escapeChar(self, char):
        replacement = 'U%05X' % ord(char)
        self.replaceCache[char] = replacement
        return replacement

    def unescapeChar(self, charcode):
        return chr(int(charcode[1:], 16))