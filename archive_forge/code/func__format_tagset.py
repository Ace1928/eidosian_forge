import re
from textwrap import wrap
from nltk.data import load
def _format_tagset(tagset, tagpattern=None):
    tagdict = load('help/tagsets/' + tagset + '.pickle')
    if not tagpattern:
        _print_entries(sorted(tagdict), tagdict)
    elif tagpattern in tagdict:
        _print_entries([tagpattern], tagdict)
    else:
        tagpattern = re.compile(tagpattern)
        tags = [tag for tag in sorted(tagdict) if tagpattern.match(tag)]
        if tags:
            _print_entries(tags, tagdict)
        else:
            print('No matching tags found.')