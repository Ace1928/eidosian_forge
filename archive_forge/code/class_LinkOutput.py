import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LinkOutput(ContainerOutput):
    """A link pointing to some destination"""
    'Or an anchor (destination)'

    def gethtml(self, link):
        """Get the HTML code for the link"""
        type = link.__class__.__name__
        if link.type:
            type = link.type
        tag = 'a class="' + type + '"'
        if link.anchor:
            tag += ' name="' + link.anchor + '"'
        if link.destination:
            link.computedestination()
        if link.url:
            tag += ' href="' + link.url + '"'
        if link.target:
            tag += ' target="' + link.target + '"'
        if link.title:
            tag += ' title="' + link.title + '"'
        return TaggedOutput().settag(tag).gethtml(link)