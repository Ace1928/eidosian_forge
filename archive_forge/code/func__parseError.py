import copy
from typing import Optional
from twisted.words.xish import domish
def _parseError(error, errorNamespace):
    """
    Parses an error element.

    @param error: The error element to be parsed
    @type error: L{domish.Element}
    @param errorNamespace: The namespace of the elements that hold the error
                           condition and text.
    @type errorNamespace: C{str}
    @return: Dictionary with extracted error information. If present, keys
             C{condition}, C{text}, C{textLang} have a string value,
             and C{appCondition} has an L{domish.Element} value.
    @rtype: C{dict}
    """
    condition = None
    text = None
    textLang = None
    appCondition = None
    for element in error.elements():
        if element.uri == errorNamespace:
            if element.name == 'text':
                text = str(element)
                textLang = element.getAttribute((NS_XML, 'lang'))
            else:
                condition = element.name
        else:
            appCondition = element
    return {'condition': condition, 'text': text, 'textLang': textLang, 'appCondition': appCondition}