import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
class token(_unicode):
    """ Represents a diffable token, generally a word that is displayed to
    the user.  Opening tags are attached to this token when they are
    adjacent (pre_tags) and closing tags that follow the word
    (post_tags).  Some exceptions occur when there are empty tags
    adjacent to a word, so there may be close tags in pre_tags, or
    open tags in post_tags.

    We also keep track of whether the word was originally followed by
    whitespace, even though we do not want to treat the word as
    equivalent to a similar word that does not have a trailing
    space."""
    hide_when_equal = False

    def __new__(cls, text, pre_tags=None, post_tags=None, trailing_whitespace=''):
        obj = _unicode.__new__(cls, text)
        if pre_tags is not None:
            obj.pre_tags = pre_tags
        else:
            obj.pre_tags = []
        if post_tags is not None:
            obj.post_tags = post_tags
        else:
            obj.post_tags = []
        obj.trailing_whitespace = trailing_whitespace
        return obj

    def __repr__(self):
        return 'token(%s, %r, %r, %r)' % (_unicode.__repr__(self), self.pre_tags, self.post_tags, self.trailing_whitespace)

    def html(self):
        return _unicode(self)