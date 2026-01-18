import json
import logging
from bs4 import BeautifulSoup
from mechanize import ParseResponseEx
from mechanize._form import AmbiguityError
from mechanize._form import ControlNotFoundError
from mechanize._form import ListControl
from urlparse import urlparse
def NoneFunc():
    return None