from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
class ScribdDocument(IFrame):
    """
    Class for embedding a Scribd document in an IPython session

    Use the start_page params to specify a starting point in the document
    Use the view_mode params to specify display type one off scroll | slideshow | book

    e.g to Display Wes' foundational paper about PANDAS in book mode from page 3

    ScribdDocument(71048089, width=800, height=400, start_page=3, view_mode="book")
    """

    def __init__(self, id, width=400, height=300, **kwargs):
        src = 'https://www.scribd.com/embeds/{0}/content'.format(id)
        super(ScribdDocument, self).__init__(src, width, height, **kwargs)