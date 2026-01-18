from a single quote by the algorithm. Therefore, a text like::
import re, sys
class smartchars(object):
    """Smart quotes and dashes
    """
    endash = '–'
    emdash = '—'
    ellipsis = '…'
    apostrophe = '’'
    quotes = {'af': '“”‘’', 'af-x-altquot': '„”‚’', 'bg': '„“‚‘', 'ca': '«»“”', 'ca-x-altquot': '“”‘’', 'cs': '„“‚‘', 'cs-x-altquot': '»«›‹', 'da': '»«›‹', 'da-x-altquot': '„“‚‘', 'de': '„“‚‘', 'de-x-altquot': '»«›‹', 'de-ch': '«»‹›', 'el': '«»“”', 'en': '“”‘’', 'en-uk-x-altquot': '‘’“”', 'eo': '“”‘’', 'es': '«»“”', 'es-x-altquot': '“”‘’', 'et': '„“‚‘', 'et-x-altquot': '«»‹›', 'eu': '«»‹›', 'fi': '””’’', 'fi-x-altquot': '»»››', 'fr': ('«\xa0', '\xa0»', '“', '”'), 'fr-x-altquot': ('«\u202f', '\u202f»', '“', '”'), 'fr-ch': '«»‹›', 'fr-ch-x-altquot': ('«\u202f', '\u202f»', '‹\u202f', '\u202f›'), 'gl': '«»“”', 'he': '”“»«', 'he-x-altquot': '„”‚’', 'hr': '„”‘’', 'hr-x-altquot': '»«›‹', 'hsb': '„“‚‘', 'hsb-x-altquot': '»«›‹', 'hu': '„”«»', 'is': '„“‚‘', 'it': '«»“”', 'it-ch': '«»‹›', 'it-x-altquot': '“”‘’', 'ja': '「」『』', 'ko': '《》〈〉', 'lt': '„“‚‘', 'lv': '„“‚‘', 'mk': '„“‚‘', 'nl': '“”‘’', 'nl-x-altquot': '„”‚’', 'nb': '«»’’', 'nn': '«»’’', 'nn-x-altquot': '«»‘’', 'no': '«»’’', 'no-x-altquot': '«»‘’', 'pl': '„”«»', 'pl-x-altquot': '«»‚’', 'pt': '«»“”', 'pt-br': '“”‘’', 'ro': '„”«»', 'ru': '«»„“', 'sh': '„”‚’', 'sh-x-altquot': '»«›‹', 'sk': '„“‚‘', 'sk-x-altquot': '»«›‹', 'sl': '„“‚‘', 'sl-x-altquot': '»«›‹', 'sq': '«»‹›', 'sq-x-altquot': '“„‘‚', 'sr': '„”’’', 'sr-x-altquot': '»«›‹', 'sv': '””’’', 'sv-x-altquot': '»»››', 'tr': '“”‘’', 'tr-x-altquot': '«»‹›', 'uk': '«»„“', 'uk-x-altquot': '„“‚‘', 'zh-cn': '“”‘’', 'zh-tw': '「」『』'}

    def __init__(self, language='en'):
        self.language = language
        try:
            self.opquote, self.cpquote, self.osquote, self.csquote = self.quotes[language.lower()]
        except KeyError:
            self.opquote, self.cpquote, self.osquote, self.csquote = '""\'\''