import datetime
import decimal
import json
from typing import Any
from itemadapter import ItemAdapter, is_item
from twisted.internet import defer
from scrapy.http import Request, Response
class ScrapyJSONDecoder(json.JSONDecoder):
    pass