import subprocess
import sys
import time
from urllib.parse import urlencode
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.linkextractors import LinkExtractor
A spider that follows all links