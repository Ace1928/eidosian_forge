import json
from itemadapter import ItemAdapter, is_item
from scrapy.contracts import Contract
from scrapy.exceptions import ContractFail
from scrapy.http import Request
class UrlContract(Contract):
    """Contract to set the url of the request (mandatory)
    @url http://scrapy.org
    """
    name = 'url'

    def adjust_request_args(self, args):
        args['url'] = self.args[0]
        return args