import re
from urllib.parse import urljoin
import bs4
from bs4.element import Comment, NavigableString, Tag
Get the text content of an element, replacing images by alt or src