import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _get_syntactic_list_within_frame(self, vnframe):
    """Returns semantics within a frame

        A utility function to retrieve semantics within a frame in VerbNet.
        Members of the syntactic dictionary:
        1) POS Tag
        2) Modifiers

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        :return: syntax_within_single_frame
        """
    syntax_within_single_frame = []
    for elt in vnframe.find('SYNTAX'):
        pos_tag = elt.tag
        modifiers = dict()
        modifiers['value'] = elt.get('value') if 'value' in elt.attrib else ''
        modifiers['selrestrs'] = [{'value': restr.get('Value'), 'type': restr.get('type')} for restr in elt.findall('SELRESTRS/SELRESTR')]
        modifiers['synrestrs'] = [{'value': restr.get('Value'), 'type': restr.get('type')} for restr in elt.findall('SYNRESTRS/SYNRESTR')]
        syntax_within_single_frame.append({'pos_tag': pos_tag, 'modifiers': modifiers})
    return syntax_within_single_frame